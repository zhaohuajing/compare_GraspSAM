| Variable       | Shape  | Type    | Units    | Meaning             |
| -------------- | ------ | ------- | -------- | ------------------- |
| `q_out`        | H×W    | float32 | unitless | grasp quality       |
| `ang_out`      | H×W    | float32 | radians  | gripper orientation |
| `w_out`        | H×W    | float32 | pixels   | gripper width       |
| `gs[i].center` | (2,)   | float   | pixels   | grasp center        |
| `gs[i].angle`  | scalar | float   | radians  | grasp rotation      |
| `gs[i].width`  | scalar | float   | pixels   | gripper opening     |
