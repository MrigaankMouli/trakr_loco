import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from mrigaank.articulation.config.trakr_cfg import TRAKR_CFG


def design_scene():
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    origins = [[0.0, 0.0, 0.0]]
    sim_utils.create_prim("/World/Origin0", "Xform", translation=origins[0])

    robot_cfg = TRAKR_CFG
    robot = Articulation(cfg=robot_cfg)

    return {"robot": robot}, origins


def run_sim(sim, entities, origins):
    robot = entities["robot"]

    idx = {name: i for i, name in enumerate(robot.joint_names)}

    LF_hip = idx["LF_hip"]
    RF_hip = idx["RF_hip"]
    LB_hip = idx["LB_hip"]
    RB_hip = idx["RB_hip"]

    LF_knee = idx["LF_knee"]
    RF_knee = idx["RF_knee"]
    LB_knee = idx["LB_knee"]
    RB_knee = idx["RB_knee"]

    sim_dt = sim.get_physics_dt()
    count = 0

    freq = 1.5          
    hip_amp = 0.35      
    knee_stance = 0.7   
    knee_swing = 1.2    

    while simulation_app.is_running():
        t = count * sim_dt
        phi = 2.0 * torch.pi * freq * t

        target = robot.data.default_joint_pos.clone()

        phase_a = torch.sin(torch.tensor(phi))           
        phase_b = torch.sin(torch.tensor(phi + torch.pi))  

        target[:, LF_hip] = hip_amp * phase_a
        target[:, RB_hip] = hip_amp * phase_a
        target[:, RF_hip] = hip_amp * phase_b
        target[:, LB_hip] = hip_amp * phase_b

        knee_a = knee_stance + (knee_swing - knee_stance) * torch.clamp(phase_a, min=0.0)
        knee_b = knee_stance + (knee_swing - knee_stance) * torch.clamp(phase_b, min=0.0)

        target[:, LF_knee] = knee_a
        target[:, RB_knee] = knee_a
        target[:, RF_knee] = knee_b
        target[:, LB_knee] = knee_b

        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        count += 1


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    entities, origins = design_scene()
    origins = torch.tensor(origins, device=sim.device)

    sim.reset()
    print("[INFO]: Simulation ready")

    run_sim(sim, entities, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()