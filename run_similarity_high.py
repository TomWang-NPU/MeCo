import os
import pickle
import logging
from glob import glob

import numpy as np
from natsort import natsorted
import argparse
import time
from rocobench import PlannedPathPolicy, LLMPathPlan, MultiArmRRT
from rocobench.envs import SortOneBlockTask, CabinetTask, MoveRopeTask, SweepTask, MakeSandwichTask, PackGroceryTask
import shutil
from rocobench.envs.base_env import MujocoSimEnv
import re


logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)


TASK_NAME_MAP = {
    "sort": SortOneBlockTask,
    "cabinet": CabinetTask,
    "rope": MoveRopeTask,
    "sweep": SweepTask,
    "sandwich": MakeSandwichTask,
    "pack": PackGroceryTask,
}


high_positions = {
    "apple": np.array([-1 ,  -1  ,  0.36]),
    "banana": np.array([-1,  -1 ,  0.25]),
    "milk": np.array([-1 ,  -1,  0.47]),
    "soda_can": np.array([-1,  -1,  0.33]),
    "bread": np.array([-1,  -1,  0.27]),
    "cereal": np.array([-1,  -1,  0.55]),
}

class SimulationRunner:
    def __init__(self, task_name: str, run_name: str = "run_1", overwrite: bool = False):

        self.task_name = task_name
        self.run_name = run_name
        base_dir = os.path.dirname(__file__)
        self.run_dir = os.path.join(base_dir, "data", "alpha", run_name)

        re_base_dir = os.path.join(base_dir, "data", "alpha")
        prefix = "re_run_"
        existing = [d for d in os.listdir(re_base_dir) if
                    os.path.isdir(os.path.join(re_base_dir, d)) and d.startswith(prefix)]

        nums = []
        for name in existing:
            match = re.match(rf"{prefix}(\d+)", name)
            if match:
                nums.append(int(match.group(1)))
        next_num = max(nums) + 1 if nums else 1
        self.re_run_dir = os.path.join(re_base_dir, f"{prefix}{next_num}")

        os.makedirs(self.run_dir, exist_ok=overwrite)
        os.makedirs(self.re_run_dir, exist_ok=True)
        self.overwrite = overwrite

        if task_name not in TASK_NAME_MAP:
            raise ValueError(f"Unknown task '{task_name}'. Please check TASK_NAME_MAP.")

        self.env = TASK_NAME_MAP[task_name](
            render_freq=2000,
            image_hw=(400, 400),
            sim_forward_steps=300,
            error_freq=30,
            error_threshold=1e-5,
            randomize_init=True,
            render_point_cloud=0,
            render_cameras=["face_panda", "face_ur5e", "teaser"],  # Fixed camera viewpoints
            one_obj_each=True,
        )

    def get_fixed_position_from_action(self, action: str):
        parts = action.split()
        if len(parts) < 2:
            return None

        object_name = parts[1]
        pos = self.env.data.body(object_name).xpos
        pos_fix = high_positions.get(object_name, None)
        pos[2] = pos_fix[2]

        return object_name, pos, self.env.data.body(object_name).xquat  # Return fixed position


    def save_simulation_state(self, step_dir: str):
        os.makedirs(step_dir, exist_ok=self.overwrite)
        sim_data = self.env.save_intermediate_state()
        with open(os.path.join(step_dir, "env_init.pkl"), "wb") as f:
            pickle.dump(sim_data, f)
        logging.info(f"Simulation state saved to {step_dir}/env_init.pkl")

    def load_simulation_state(self, step_dir: str):
        env_init_fname = os.path.join(step_dir, "env_init.pkl")
        if not os.path.exists(env_init_fname):
            logging.error(f" {env_init_fname} dismiss")
            return False

        with open(env_init_fname, "rb") as f:
            saved_data = pickle.load(f)
            self.env.load_saved_state(saved_data)
        logging.info(f"Loaded simulation state from {env_init_fname}")
        return True

    def par_llm_plan(self, step_dir: str):

        pkl_path = os.path.join(step_dir, "llm_plan_0.pkl")

        if not os.path.exists(pkl_path):
            logging.error(f"File {pkl_path} is missing.")
            return {}

        with open(pkl_path, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                logging.error(f"dismiss {pkl_path}: {e}")
                return {}

        if not hasattr(data, 'action_strs') or not isinstance(data.action_strs, dict):
            logging.error("llm_plan_0.pkl does not contain a valid action_strs dict.")
            return {}
        return data.action_strs

    def modify_llm_plan(self, step_dir: str, re_step_dir: str, type: str, pose ):
        input_file_path = os.path.join(step_dir, "llm_plan_0.pkl")
        output_file_path = os.path.join(re_step_dir, "llm_plan_0.pkl")

        if not os.path.exists(input_file_path):
            logging.error(f"file {input_file_path} dismiss")
            return False

        with open(input_file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except Exception as e:
                logging.error(f"dismiss {input_file_path}: {e}")
                return False

        new_values = pose


        data.ee_targets[type][:3] = new_values

        with open(output_file_path, "wb") as file:
            pickle.dump(data, file)

        logging.info(f"changed llm_plan_0.pkl save {output_file_path}")
        return True

    def modify_llm_plan_1(self, step_dir: str, re_step_dir: str, type: str, pose ):
        input_file_path = os.path.join(step_dir, "llm_plan_1.pkl")
        output_file_path = os.path.join(re_step_dir, "llm_plan_1.pkl")

        if not os.path.exists(input_file_path):
            logging.error(f"File {input_file_path} is missing.")
            return False

        with open(input_file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except Exception as e:
                logging.error(f"dismiss {input_file_path}: {e}")
                return False

        new_values = pose


        data.ee_targets[type][:3] = new_values

        with open(output_file_path, "wb") as file:
            pickle.dump(data, file)

        logging.info(f"Updated llm_plan_1.pkl saved to {output_file_path}")
        return True

    def modify_llm_plan_all(self, step_dir: str, re_step_dir: str, type_0: str, pose_0, way_0,  type_1: str, pose_1, way_1):

        input_file_path = os.path.join(step_dir, "llm_plan_0.pkl")
        output_file_path = os.path.join(re_step_dir, "llm_plan_0.pkl")

        if not os.path.exists(input_file_path):
            logging.error(f"{input_file_path} dismiss")
            return False

        with open(input_file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except Exception as e:
                logging.error(f"dismiss {input_file_path}: {e}")
                return False

        # Update the first three values
        new_values = pose_0
        data.ee_targets[type_0][:3] = new_values[:3]

        new_values = pose_1
        data.ee_targets[type_1][:3] = new_values[:3]

        data.ee_waypoints[type_0] = [way_0]
        data.ee_waypoints[type_1] = [way_1]

        # Save the modified plan to re_step_dir
        with open(output_file_path, "wb") as file:
            pickle.dump(data, file)

        logging.info(f"changed llm_plan_0.pkl saved {output_file_path}")
        return True

    def modify_llm_plan_all_pack(self, step_dir: str, re_step_dir: str, type_0: str, pose_0, type_1: str, pose_1, fk, selected_indices):
        input_file_path = os.path.join(step_dir, "llm_plan_0.pkl")
        output_file_path = os.path.join(re_step_dir, "llm_plan_0.pkl")

        if not os.path.exists(input_file_path):
            logging.error(f"file {input_file_path} dismiss")
            return False

        with open(input_file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except Exception as e:
                logging.error(f"no {input_file_path}: {e}")
                return False

        new_values = pose_0
        data.ee_targets[type_0][:3] = new_values[:3]

        new_values = pose_1
        data.ee_targets[type_1][:3] = new_values[:3]

        # Initialize ee_waypoints
        data.ee_waypoints[type_0] = []
        data.ee_waypoints[type_1] = []

        # Append selected FK waypoints to ee_waypoints
        for idx in selected_indices:
            data.ee_waypoints[type_0].append(fk[idx][type_0])
            data.ee_waypoints[type_1].append(fk[idx][type_1])


        # Save the modified plan to re_step_dir
        with open(output_file_path, "wb") as file:
            pickle.dump(data, file)

        logging.info(f"changed llm_plan_0.pkl save {output_file_path}")
        return True


    def modify_llm_plan_all_ee(self, step_dir: str, re_step_dir: str, type_0: str, p_p_0, pose_0, pose_2, pos_4,pos_6, type_1: str, p_p_1, pose_1, pose_3, pos_5,pos_7):
        input_file_path = os.path.join(step_dir, "llm_plan_0.pkl")
        output_file_path = os.path.join(re_step_dir, "llm_plan_0.pkl")

        if not os.path.exists(input_file_path):
            logging.error(f"file {input_file_path} dismiss")
            return False

        with open(input_file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except Exception as e:
                logging.error(f"no {input_file_path}: {e}")
                return False

        # Update ee_waypoints
        data.ee_waypoints[type_0] = [p_p_0, pose_0, pose_2, pos_4, pos_6]
        data.ee_waypoints[type_1] = [p_p_1, pose_1, pose_3, pos_5, pos_7]

        # Save the modified plan to re_step_dir
        with open(output_file_path, "wb") as file:
            pickle.dump(data, file)

        logging.info(f"change llm_plan_0.pkl save {output_file_path}")
        return True

    def modify_llm_plan_all_place(self, step_dir: str, re_step_dir: str, type_0: str, type_1: str, fk, selected_indices):
        input_file_path = os.path.join(step_dir, "llm_plan_0.pkl")
        output_file_path = os.path.join(re_step_dir, "llm_plan_0.pkl")


        with open(input_file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except Exception as e:
                logging.error(f"no {input_file_path}: {e}")
                return False

        data.ee_waypoints[type_0] = []
        data.ee_waypoints[type_1] = []

        for idx in selected_indices:
            data.ee_waypoints[type_0].append(fk[idx][type_0])
            data.ee_waypoints[type_1].append(fk[idx][type_1])

        with open(output_file_path, "wb") as file:
            pickle.dump(data, file)

        logging.info(f"llm_plan_0.pkl save {output_file_path}")
        return True

    def modify_llm_plan_all_1(self, step_dir: str, re_step_dir: str, type_0: str, pose_0,  type_1: str, pose_1):

        input_file_path = os.path.join(step_dir, "llm_plan_0.pkl")
        output_file_path = os.path.join(re_step_dir, "llm_plan_0.pkl")


        with open(input_file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except Exception as e:
                logging.error(f"no {input_file_path}: {e}")
                return False

        # Update the first three values
        new_values = pose_0
        data.ee_targets[type_0][:7] = new_values

        new_values = pose_1
        data.ee_targets[type_1][:7] = new_values

        temp_list = list(data.tograsp[type_0])
        temp_list[2] = 1
        data.tograsp[type_0] = tuple(temp_list)
        temp_list = list(data.tograsp[type_1])
        temp_list[2] = 1
        data.tograsp[type_1] = tuple(temp_list)

        data.return_home[type_0] = False
        data.return_home[type_1] = False

        with open(output_file_path, "wb") as file:
            pickle.dump(data, file)

        return True

    def modify_llm_plan_all_2(self, step_dir: str, re_step_dir: str, type_0: str, pose_0,  type_1: str, pose_1):

        input_file_path = os.path.join(step_dir, "llm_plan_0.pkl")
        output_file_path = os.path.join(re_step_dir, "llm_plan_0.pkl")

        with open(input_file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except Exception as e:
                logging.error(f"no {input_file_path}: {e}")
                return False


        new_values = pose_0
        data.ee_targets[type_0][3:7] = new_values[3:7]

        new_values = pose_1
        data.ee_targets[type_1][3:7] = new_values[3:7]

        with open(output_file_path, "wb") as file:
            pickle.dump(data, file)

        logging.info(f"llm_plan_0.pkl save {output_file_path}")
        return True

    def compute_fk_from_rrt(self, step_dir: str):

        pkl_path = os.path.join(step_dir, "rrt_plan_0.pkl")

        if not os.path.exists(pkl_path):
            logging.error(f"Missing {pkl_path}; cannot compute FK.")
            return

        with open(pkl_path, "rb") as f:
            try:
                rrt_plan = pickle.load(f)
            except Exception as e:
                logging.error(f"no {pkl_path}: {e}")
                return

        if not isinstance(rrt_plan, list) or len(rrt_plan) == 0:
            logging.error(f"rrt_plan_0.pkl error")
            return

        qpos_targets = [qpos for qpos in rrt_plan if isinstance(qpos, (list, tuple)) or hasattr(qpos, "shape")]

        if not qpos_targets:
            logging.error("no qpos_target data")
            return

        return qpos_targets

    def execute_plan(self, step_dir: str, flag: int):
        action_files = natsorted(glob(os.path.join(step_dir, "actions_*.pkl")))

        if not action_files:
            logging.warning(f"{step_dir} no action file")
            return False

        plan_files = action_files

        if flag==-1:
            flag=len(plan_files)-1

        plan_file = action_files[flag]


        with open(plan_file, "rb") as f:
            actions = pickle.load(f)

        num_steps = 0
        for sim_action in actions:
            obs, reward, done, info = self.env.step(sim_action, verbose=False)
            num_steps += 1

        if num_steps > 0 and flag==len(plan_files)-1:
            vid_name = os.path.join(step_dir, "execute_2.mp4")
            self.env.export_render_to_video(vid_name, out_type="mp4", fps=50)  # Export video (no per-call camera override)
            logging.info(f"Execution video saved to {vid_name}")

        return True

    def rrt_replan(self, step_dir: str, re_step_dir: str):

        logging.info(f"Performing RRT replanning for {re_step_dir}")

        llm_plan_files = natsorted(glob(os.path.join(re_step_dir, "llm_plan_*.pkl")))
        if not llm_plan_files:
            logging.warning(f"No LLM plan found in {re_step_dir}, using fallback planning.")
            return False


        llm_plans = []
        for plan_file in llm_plan_files:
            with open(plan_file, "rb") as f:
                llm_plans.append(pickle.load(f))

        logging.info(f"Loaded {len(llm_plans)} LLM plans from {re_step_dir}.")


        num_plans = len(llm_plans)

        for i, llm_plan in enumerate(llm_plans):


            policy = PlannedPathPolicy(
                physics=self.env.physics,
                robots=self.env.get_sim_robots(),
                path_plan=llm_plan,
                graspable_object_names=self.env.get_graspable_objects(),
                allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                plan_splitted=True
            )


            logging.info(f"Running RRT planner for {re_step_dir}, plan {i}.")
            plan_success, reason = policy.plan(self.env)
            logging.info(f"RRT Planning success: {plan_success}, Reason: {reason}")

            if not plan_success:
                logging.error(f"RRT planning failed: no path found for plan {i}.")
                continue

            actions_path = os.path.join(re_step_dir, f"actions_{i}.pkl")
            with open(actions_path, "wb") as f:
                pickle.dump(policy.action_buffer, f)
            logging.info(f"RRT save {actions_path}")

            if i != num_plans-1 and num_plans != 1:
                success = self.execute_plan(re_step_dir, i)


        return True

    def get_pos(self, step_dir: str):

        logging.info(f"Performing RRT replanning for {step_dir}")

        llm_plan_files = natsorted(glob(os.path.join(step_dir, "llm_plan_0.pkl")))
        if not llm_plan_files:
            logging.warning(f"No LLM plan found in {step_dir}, using fallback planning.")
            return False


        llm_plans = []
        for plan_file in llm_plan_files:
            with open(plan_file, "rb") as f:
                llm_plans.append(pickle.load(f))

        logging.info(f"Loaded {len(llm_plans)} LLM plans from {step_dir}.")
        policy = PlannedPathPolicy(
                physics=self.env.physics,
                robots=self.env.get_sim_robots(),
                path_plan=llm_plans[0],
                graspable_object_names=self.env.get_graspable_objects(),
                allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
        )

        rrt_plan = self.compute_fk_from_rrt(step_dir)

        longest_min, longest_max = 0, 0
        current_min = 0
        max_length = 0
        in_collision = False

        for i, qpos in enumerate(rrt_plan):
            flag = policy.collision(self.env, np.array(qpos))
            if flag:
                if not in_collision:
                    current_length = i - current_min
                    if current_length > max_length:
                        max_length = current_length
                        longest_min, longest_max = current_min, i - 1
                in_collision = True
            else:
                if in_collision:
                    current_min = i
                in_collision = False


        if not in_collision and (len(rrt_plan) - current_min > max_length):
            longest_min, longest_max = current_min, len(rrt_plan) - 1

        fk_results = []
        for i in range(longest_min, longest_max + 1):
            qpos = rrt_plan[i]
            fk_result = policy.planner(self.env, np.array(qpos))
            fk_results.append(fk_result)
        return fk_results



    def select_indices(self, boundary_left, boundary_right, epsilon, rounding):


        pts = np.linspace(boundary_left, boundary_right, epsilon + 2)[1:-1]
        if rounding == 'ceil':
            pts = np.ceil(pts)
        elif rounding == 'floor':
            pts = np.floor(pts)
        return pts.astype(int).tolist()

    def run(self, start_step: int = 0):

        step_dirs = natsorted(glob(os.path.join(self.run_dir, "step_*")))

        os.makedirs(self.re_run_dir, exist_ok=True)


        doub = np.zeros(len(step_dirs))
        j = start_step
        while j < len(step_dirs):
            step_dir=step_dirs[j]
            re_step_dir = os.path.join(self.re_run_dir, os.path.basename(step_dir))
            os.makedirs(re_step_dir, exist_ok=True)

            self.save_simulation_state(re_step_dir)
            if not self.load_simulation_state(re_step_dir):
                continue


            llm_actions = self.par_llm_plan(step_dir)


            need_replan = True

            robot0_name = None
            robot0_action = None
            robot1_name = None
            robot1_action = None
            i = 0
            for robot, action in llm_actions.items():
                if i == 0:
                    robot0_name = robot
                    robot0_action = action
                else:
                    robot1_name = robot
                    robot1_action = action
                i=i+1

            object_name_0, pos_0, quat_0 = self.get_fixed_position_from_action(robot0_action)
            object_name_1, pos_1, quat_1 = self.get_fixed_position_from_action(robot1_action)
            combined_0 = np.concatenate([np.array(pos_0), np.array(quat_0)])
            combined_1 = np.concatenate([np.array(pos_1), np.array(quat_1)])

            if "PICK" in robot0_action and "PICK" in robot1_action:
                fk = self.get_pos(step_dir)
                filtered_fk = []
                flag = fk[0]
                for item in fk:
                    alice_x = item[robot0_name][0]
                    bob_x = item[robot1_name][0]
                    if combined_0[0] <= alice_x and combined_1[0] <= bob_x:
                        filtered_fk.append(item)
                        flag = item
                    if combined_0[0] > alice_x and combined_1[0] <= bob_x:
                        item[robot0_name] = flag[robot0_name]
                        filtered_fk.append(item)
                    if combined_0[0] <= alice_x and combined_1[0] > bob_x:
                        item[robot1_name] = flag[robot1_name]
                        filtered_fk.append(item)

                fk = filtered_fk


                print(len(fk))
                pos_0_idx = max(range(len(fk)), key=lambda i: fk[i][robot0_name][2])
                pos_1_idx = max(range(len(fk)), key=lambda i: fk[i][robot1_name][2])
                print(pos_0_idx, pos_1_idx)

                epsilon = 5

                if pos_0_idx >= 0:
                    left_indices_0 = self.select_indices(0, pos_0_idx, epsilon, 'ceil')
                    right_indices_0 = self.select_indices(pos_0_idx, len(fk) - 1, epsilon, 'floor')
                else:
                    left_indices_0, right_indices_0 = [], []

                if pos_1_idx >= 0:
                    left_indices_1 = self.select_indices(0, pos_1_idx, epsilon, 'ceil')
                    right_indices_1 = self.select_indices(pos_1_idx, len(fk) - 1, epsilon, 'floor')
                else:
                    left_indices_1, right_indices_1 = [], []

                end = len(filtered_fk) - 1
                b_0 = (filtered_fk[end][robot0_name][0] - filtered_fk[0][robot0_name][0]) / (combined_0[0] - filtered_fk[0][robot0_name][0])
                b_1 = (filtered_fk[end][robot1_name][0] - filtered_fk[0][robot1_name][0]) / (combined_1[0] - filtered_fk[0][robot1_name][0])
                print(b_0, b_1)




                selected_indices = sorted(
                    set(left_indices_0 + right_indices_0 + left_indices_1 + right_indices_1 + [pos_0_idx] + [pos_1_idx]))
                print(selected_indices)
                self.modify_llm_plan_all_pack(step_dir, re_step_dir, robot0_name, combined_0, robot1_name, combined_1, fk, selected_indices)

            elif "PLACE" in robot1_action and "PLACE" in robot0_action:
                fk = self.get_pos(step_dir)
                filtered_fk = []
                for item in fk:
                    alice_x = item[robot0_name][0]
                    bob_x = item[robot1_name][0]
                    if combined_0[0] <= alice_x and combined_1[0] <= bob_x:
                        filtered_fk.append(item)
                if len(filtered_fk)>0:
                    flag = filtered_fk[0]
                else:
                    break

                filtered_fk = []
                for item in fk:
                    alice_x = item[robot0_name][0]
                    bob_x = item[robot1_name][0]
                    if combined_0[0] <= alice_x and combined_1[0] <= bob_x:
                        filtered_fk.append(item)
                        flag = item
                    if combined_0[0] > alice_x and combined_1[0] <= bob_x:
                        item[robot0_name] = flag[robot0_name]
                        filtered_fk.append(item)
                    if combined_0[0] <= alice_x and combined_1[0] > bob_x:
                        item[robot1_name] = flag[robot1_name]
                        filtered_fk.append(item)
                pkl_path = os.path.join(step_dir, "env_end.pkl")

                if not os.path.exists(pkl_path):
                    logging.error(f"no {pkl_path} !")
                    return {}

                with open(pkl_path, "rb") as f:
                    try:
                        data = pickle.load(f)
                    except Exception as e:
                        logging.error(f"no {pkl_path}: {e}")

                pos_0, quat_0 = data.env_state.objects[object_name_0].xpos, data.env_state.objects[object_name_0].xquat
                pos_1, quat_1 = data.env_state.objects[object_name_1].xpos, data.env_state.objects[object_name_1].xquat
                end = len(filtered_fk) - 1
                b_0 = (filtered_fk[end][robot0_name][0] - filtered_fk[0][robot0_name][0]) / (pos_0[0] - combined_0[0])
                b_1 = (filtered_fk[end][robot1_name][0] - filtered_fk[0][robot1_name][0]) / (pos_1[0] - combined_1[0])
                print(b_0, b_1)





                fk = filtered_fk
                print(len(fk))
                pos_0_idx = max(range(len(fk)), key=lambda i: fk[i][robot0_name][2])
                pos_1_idx = max(range(len(fk)), key=lambda i: fk[i][robot1_name][2])
                print(pos_0_idx, pos_1_idx)
                epsilon = 100

                if pos_0_idx >= 0:
                    left_indices_0 = self.select_indices(0, pos_0_idx, epsilon, 'ceil')
                    right_indices_0 = self.select_indices(pos_0_idx, len(fk)-1, epsilon, 'floor')
                else:
                    left_indices_0, right_indices_0 = [], []

                if pos_1_idx >= 0:
                    left_indices_1 = self.select_indices(0, pos_1_idx, epsilon, 'ceil')
                    right_indices_1 = self.select_indices(pos_1_idx, len(fk)-1, epsilon, 'floor')
                else:
                    left_indices_1, right_indices_1 = [], []

                # Merge, deduplicate, and sort
                selected_indices = sorted(set(left_indices_0 + right_indices_0 + left_indices_1 + right_indices_1 + [pos_0_idx] + [pos_1_idx]))
                print(selected_indices)
                self.modify_llm_plan_all_place(step_dir, re_step_dir, robot0_name, robot1_name, fk, selected_indices)

            if need_replan:
                logging.info(f"Step {step_dir}: No direct interaction found, calling rrt_replan.")
                self.rrt_replan(step_dir, re_step_dir)

            j=j+1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help="tsak")
    parser.add_argument("--run_name", type=str, default="run_1", help="file")
    parser.add_argument("--start_step", type=int, default=0, help="start step")

    args = parser.parse_args()

    runner = SimulationRunner(task_name=args.task_name, run_name=args.run_name, overwrite=True)
    runner.run(start_step=args.start_step)



