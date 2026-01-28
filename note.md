| Variable       | Shape  | Type    | Units    | Meaning             |
| -------------- | ------ | ------- | -------- | ------------------- |
| `q_out`        | H×W    | float32 | unitless | grasp quality       |
| `ang_out`      | H×W    | float32 | radians  | gripper orientation |
| `w_out`        | H×W    | float32 | pixels   | gripper width       |
| `gs[i].center` | (2,)   | float   | pixels   | grasp center        |
| `gs[i].angle`  | scalar | float   | radians  | grasp rotation      |
| `gs[i].width`  | scalar | float   | pixels   | gripper opening     |




After ./run_docker.sh

(GraspSAM) root@ccc99bfd3bce:~/graspnet_ws/src# history
    2  source /opt/conda/etc/profile.d/conda.sh &&     conda activate GraspSAM &&     python -m pip install --upgrade pip &&     python -m pip install --no-cache-dir       --index-url https://download.pytorch.org/whl/cu117       torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
    3  python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
import groundingdino
from groundingdino.models import build_model
print("GroundingDINO OK")
PY

    4  conda run -n GraspSAM python -m pip uninstall -y groundingdino || true
    5  cd ~/graspnet_ws/src/GraspSam_ros2/compare_GraspSAM/GroundingDINO
    6  # install in editable mode (so you can keep editing the repo)
    7  conda run -n GraspSAM python -m pip install -e .
    8  # rebuild the CUDA extension in-place (ensures the .so is present in the package dir)
    9  conda run -n GraspSAM python setup.py build_ext --inplace
   10  cd ~/graspnet_ws/src/graspsam_ros2/compare_GraspSAM/GroundingDINO
   13  pip install -e .
   15  strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
   16  apt update
   17  apt install -y libstdc++6
   18  strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
   19  python -c "from groundingdino.models import build_model"
   22  conda install -y -c conda-forge libstdcxx-ng libgcc-ng
   23  strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29
   24  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
   25  python -c "from groundingdino.models import build_model; print('build_model OK')"

   27  apt-get update && apt-get install -y     libgl1     libglib2.0-0     && rm -rf /var/lib/apt/lists/*
   28  python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
from groundingdino.models import build_model
print("GroundingDINO build_model OK")
PY

   29  liapt-get update || true
   30  apt-get install -y ca-certificates
   31  rm -f /etc/apt/sources.list.d/cuda.list       /etc/apt/sources.list.d/nvidia-ml.list || true
   32  apt-get update
   33  apt-get install -y     libgl1     libglib2.0-0
   34  ldconfig -p | grep libGL.so.1
   35  python - <<'PY'
from groundingdino.models import build_model
print("GroundingDINO build_model OK")
PY

   36  cd ..
   37  python eval.py --root ./datasets/Jacquard_Samples/Samples/1a9fa4c269cfcc1b738e43095496b061/ --ckp_path ./pretrained_checkpoint/sam_vit_b_01ec64.pth --sam-encoder-type vit_b
   38  docker exec -it graspsam_dev bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate GraspSAM && echo CONDA_PREFIX=$CONDA_PREFIX && strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29 || echo "NO 3.4.29 in conda libstdc++"'


root@495346281fc8:~/graspnet_ws/src/graspsam_ros2/graspsam_docker# ls /media/Grasp-Anything/scene_description/
Display all 994860 possibilities? (y or n)

root@495346281fc8:~/graspnet_ws/src/graspsam_ros2/graspsam_docker# ls /media/Grasp-Anything/mask/
Display all 1872664 possibilities? (y or n)