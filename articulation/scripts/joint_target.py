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

import time

def run_sim(sim, entities, origins):
    robot = entities["robot"]

    print(robot.joint_names)
    print(robot.num_joints)

    target = robot.data.default_joint_pos.clone()

    target[:,6] = 0.5

    robot.set_joint_position_target(target)
    robot.write_data_to_sim()

    print("[INFO]: Command sent.")

    while simulation_app.is_running():
        sim.step()
        robot.update(sim.get_physics_dt())


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