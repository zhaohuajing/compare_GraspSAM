| Variable       | Shape  | Type    | Units    | Meaning             |
| -------------- | ------ | ------- | -------- | ------------------- |
| `q_out`        | H×W    | float32 | unitless | grasp quality       |
| `ang_out`      | H×W    | float32 | radians  | gripper orientation |
| `w_out`        | H×W    | float32 | pixels   | gripper width       |
| `gs[i].center` | (2,)   | float   | pixels   | grasp center        |
| `gs[i].angle`  | scalar | float   | radians  | grasp rotation      |
| `gs[i].width`  | scalar | float   | pixels   | gripper opening     |


(GraspSAM) root@2141f3013f31:~/graspnet_ws/src/graspsam_ros2/compare_GraspSAM_backup_250112# python train.py --root /media/Jacquard_v2/ --save --sam-encoder-type vit_t
len jaccquard: 7333
len jaccquard: 7333
train_dataset size : 7333
default sam encoder is loading
default sam decoder is loading
--------------------------------------------------------------------------------
Epoch: [0][   0/1833]  Loss 5.6541 (5.6541)  pos_loss 0.6772 (0.6772)  cos_loss 0.7319 (0.7319)  sin_loss 0.2479 (0.2479)  width_loss 0.3929 (0.3929)  mask_loss 1.5543 (1.5543)
Epoch: [0][  50/1833]  Loss 3.7148 (4.1984)  pos_loss 0.3249 (0.3284)  cos_loss 0.5721 (0.5882)  sin_loss 0.2666 (0.3313)  width_loss 0.0420 (0.1135)  mask_loss 1.3035 (1.4756)
Epoch: [0][ 100/1833]  Loss 3.6448 (3.8144)  pos_loss 0.2307 (0.2889)  cos_loss 0.7668 (0.5624)  sin_loss 0.2472 (0.3117)  width_loss 0.0369 (0.0780)  mask_loss 1.0816 (1.3324)
Epoch: [0][ 150/1833]  Loss 3.2106 (3.6189)  pos_loss 0.1805 (0.2656)  cos_loss 0.4839 (0.5548)  sin_loss 0.2664 (0.3075)  width_loss 0.0386 (0.0661)  mask_loss 1.2718 (1.2308)
Epoch: [0][ 200/1833]  Loss 2.2111 (3.4547)  pos_loss 0.2341 (0.2509)  cos_loss 0.3638 (0.5450)  sin_loss 0.2211 (0.3078)  width_loss 0.0349 (0.0609)  mask_loss 0.5033 (1.1256)
Epoch: [0][ 250/1833]  Loss 2.3920 (3.2664)  pos_loss 0.1633 (0.2407)  cos_loss 0.4922 (0.5412)  sin_loss 0.3085 (0.3053)  width_loss 0.0351 (0.0571)  mask_loss 0.3936 (0.9778)
Epoch: [0][ 300/1833]  Loss 2.5620 (3.1022)  pos_loss 0.2012 (0.2345)  cos_loss 0.6297 (0.5349)  sin_loss 0.3007 (0.3034)  width_loss 0.0432 (0.0541)  mask_loss 0.2126 (0.8484)
Epoch: [0][ 350/1833]  Loss 2.0067 (2.9766)  pos_loss 0.1989 (0.2300)  cos_loss 0.3908 (0.5281)  sin_loss 0.2874 (0.3013)  width_loss 0.0244 (0.0515)  mask_loss 0.2037 (0.7548)
Epoch: [0][ 400/1833]  Loss 2.2812 (2.8811)  pos_loss 0.1775 (0.2255)  cos_loss 0.5732 (0.5264)  sin_loss 0.2923 (0.2985)  width_loss 0.0302 (0.0492)  mask_loss 0.1351 (0.6819)
Epoch: [0][ 450/1833]  Loss 2.4094 (2.8012)  pos_loss 0.1804 (0.2220)  cos_loss 0.7712 (0.5233)  sin_loss 0.1771 (0.2967)  width_loss 0.0409 (0.0468)  mask_loss 0.0703 (0.6236)
Epoch: [0][ 500/1833]  Loss 2.0449 (2.7388)  pos_loss 0.1865 (0.2190)  cos_loss 0.4437 (0.5206)  sin_loss 0.2889 (0.2974)  width_loss 0.0187 (0.0448)  mask_loss 0.1692 (0.5752)
Epoch: [0][ 550/1833]  Loss 1.6653 (2.6861)  pos_loss 0.1610 (0.2160)  cos_loss 0.3016 (0.5170)  sin_loss 0.2141 (0.2988)  width_loss 0.0139 (0.0429)  mask_loss 0.2842 (0.5368)
Epoch: [0][ 600/1833]  Loss 1.7692 (2.6399)  pos_loss 0.1607 (0.2132)  cos_loss 0.4133 (0.5155)  sin_loss 0.2238 (0.2989)  width_loss 0.0126 (0.0411)  mask_loss 0.1483 (0.5025)
Epoch: [0][ 650/1833]  Loss 2.1551 (2.5979)  pos_loss 0.2153 (0.2110)  cos_loss 0.5565 (0.5138)  sin_loss 0.2491 (0.2983)  width_loss 0.0254 (0.0395)  mask_loss 0.0626 (0.4729)
Epoch: [0][ 700/1833]  Loss 2.4208 (2.5657)  pos_loss 0.1789 (0.2089)  cos_loss 0.5795 (0.5146)  sin_loss 0.3886 (0.2968)  width_loss 0.0249 (0.0381)  mask_loss 0.0770 (0.4490)
Epoch: [0][ 750/1833]  Loss 2.1140 (2.5354)  pos_loss 0.1919 (0.2070)  cos_loss 0.5403 (0.5134)  sin_loss 0.2170 (0.2965)  width_loss 0.0206 (0.0367)  mask_loss 0.1743 (0.4284)
Epoch: [0][ 800/1833]  Loss 2.0253 (2.5051)  pos_loss 0.1787 (0.2050)  cos_loss 0.6704 (0.5118)  sin_loss 0.1260 (0.2955)  width_loss 0.0146 (0.0354)  mask_loss 0.0459 (0.4094)
Epoch: [0][ 850/1833]  Loss 2.0495 (2.4826)  pos_loss 0.1590 (0.2034)  cos_loss 0.5177 (0.5122)  sin_loss 0.3002 (0.2949)  width_loss 0.0176 (0.0343)  mask_loss 0.0607 (0.3930)
Epoch: [0][ 900/1833]  Loss 2.2395 (2.4598)  pos_loss 0.1807 (0.2021)  cos_loss 0.6447 (0.5116)  sin_loss 0.2313 (0.2942)  width_loss 0.0133 (0.0334)  mask_loss 0.0994 (0.3774)
Epoch: [0][ 950/1833]  Loss 2.3166 (2.4396)  pos_loss 0.2445 (0.2010)  cos_loss 0.4990 (0.5099)  sin_loss 0.3351 (0.2944)  width_loss 0.0200 (0.0325)  mask_loss 0.1193 (0.3638)
Epoch: [0][1000/1833]  Loss 1.7316 (2.4207)  pos_loss 0.1639 (0.2003)  cos_loss 0.3548 (0.5092)  sin_loss 0.2175 (0.2931)  width_loss 0.0128 (0.0317)  mask_loss 0.2338 (0.3521)
Epoch: [0][1050/1833]  Loss 1.8657 (2.4034)  pos_loss 0.1808 (0.1991)  cos_loss 0.4872 (0.5078)  sin_loss 0.2248 (0.2932)  width_loss 0.0187 (0.0310)  mask_loss 0.0429 (0.3410)
Epoch: [0][1100/1833]  Loss 2.1748 (2.3872)  pos_loss 0.1786 (0.1980)  cos_loss 0.4610 (0.5064)  sin_loss 0.3835 (0.2931)  width_loss 0.0185 (0.0304)  mask_loss 0.0916 (0.3313)
Epoch: [0][1150/1833]  Loss 2.5201 (2.3727)  pos_loss 0.2037 (0.1969)  cos_loss 0.6624 (0.5067)  sin_loss 0.3045 (0.2921)  width_loss 0.0224 (0.0298)  mask_loss 0.1339 (0.3219)
Epoch: [0][1200/1833]  Loss 1.8398 (2.3585)  pos_loss 0.1599 (0.1958)  cos_loss 0.4427 (0.5055)  sin_loss 0.2780 (0.2921)  width_loss 0.0176 (0.0292)  mask_loss 0.0434 (0.3132)
Epoch: [0][1250/1833]  Loss 1.8593 (2.3443)  pos_loss 0.1806 (0.1950)  cos_loss 0.4214 (0.5039)  sin_loss 0.2493 (0.2920)  width_loss 0.0122 (0.0287)  mask_loss 0.1326 (0.3051)
Epoch: [0][1300/1833]  Loss 1.7576 (2.3297)  pos_loss 0.1563 (0.1942)  cos_loss 0.4649 (0.5024)  sin_loss 0.1909 (0.2915)  width_loss 0.0173 (0.0282)  mask_loss 0.0990 (0.2970)
Epoch: [0][1350/1833]  Loss 2.2504 (2.3164)  pos_loss 0.2350 (0.1933)  cos_loss 0.5312 (0.5007)  sin_loss 0.3050 (0.2910)  width_loss 0.0162 (0.0277)  mask_loss 0.0754 (0.2907)
Epoch: [0][1400/1833]  Loss 2.3154 (2.3047)  pos_loss 0.1477 (0.1925)  cos_loss 0.5638 (0.4990)  sin_loss 0.3276 (0.2911)  width_loss 0.0148 (0.0273)  mask_loss 0.2075 (0.2848)
Epoch: [0][1450/1833]  Loss 1.9090 (2.2964)  pos_loss 0.1969 (0.1919)  cos_loss 0.4511 (0.4986)  sin_loss 0.2443 (0.2915)  width_loss 0.0168 (0.0270)  mask_loss 0.0907 (0.2784)
Epoch: [0][1500/1833]  Loss 1.8531 (2.2854)  pos_loss 0.1941 (0.1914)  cos_loss 0.3895 (0.4973)  sin_loss 0.2767 (0.2913)  width_loss 0.0184 (0.0266)  mask_loss 0.0957 (0.2724)
Epoch: [0][1550/1833]  Loss 2.1967 (2.2747)  pos_loss 0.1611 (0.1906)  cos_loss 0.5988 (0.4963)  sin_loss 0.2898 (0.2909)  width_loss 0.0167 (0.0262)  mask_loss 0.0638 (0.2667)
Epoch: [0][1600/1833]  Loss 1.8008 (2.2659)  pos_loss 0.1534 (0.1901)  cos_loss 0.4228 (0.4949)  sin_loss 0.2821 (0.2916)  width_loss 0.0144 (0.0259)  mask_loss 0.0552 (0.2609)
Epoch: [0][1650/1833]  Loss 2.0550 (2.2573)  pos_loss 0.1492 (0.1895)  cos_loss 0.4267 (0.4939)  sin_loss 0.3515 (0.2916)  width_loss 0.0157 (0.0256)  mask_loss 0.1689 (0.2560)
Epoch: [0][1700/1833]  Loss 2.2862 (2.2486)  pos_loss 0.1512 (0.1891)  cos_loss 0.6830 (0.4926)  sin_loss 0.2471 (0.2917)  width_loss 0.0168 (0.0254)  mask_loss 0.0901 (0.2510)
Epoch: [0][1750/1833]  Loss 1.9799 (2.2403)  pos_loss 0.1493 (0.1886)  cos_loss 0.4844 (0.4918)  sin_loss 0.2697 (0.2914)  width_loss 0.0154 (0.0251)  mask_loss 0.1423 (0.2466)
Epoch: [0][1800/1833]  Loss 1.8107 (2.2330)  pos_loss 0.1293 (0.1880)  cos_loss 0.4260 (0.4914)  sin_loss 0.2217 (0.2907)  width_loss 0.0217 (0.0248)  mask_loss 0.2133 (0.2431)
--------------------------------------------------------------------------------
save model in final_result/total_vit_t_default/jacquard/2026-02-28-04-40-49
Epoch: [1][   0/1833]  Loss 2.3089 (2.3089)  pos_loss 0.1664 (0.1664)  cos_loss 0.4744 (0.4744)  sin_loss 0.3834 (0.3834)  width_loss 0.0151 (0.0151)  mask_loss 0.2302 (0.2302)
