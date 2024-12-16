# Simple CPU final frame
![image](/1.jpg)

His:

[Taichi] version 1.7.2, llvm 15.0.4, commit 0131dce9, linux, python 3.10.12\
[Taichi] Starting on arch=x64\
Initial k_spring: [0.9225717  2.6130235  3.528798   4.332724   3.9298193  0.39066303\
 1.7221198  0.4286768  1.9457561  2.2746367  4.820588   4.624189\
 4.4663568  2.451238   0.16982393 3.99396    2.8387103  2.892236\
 0.9569663  4.297811  ]\
Epoch 100, Loss: 4646.0986\
Epoch 200, Loss: 4646.0957\
Epoch 300, Loss: 4646.0928\
Epoch 400, Loss: 4646.0894\
Epoch 500, Loss: 4646.0854\
Epoch 600, Loss: 4646.0820\
Epoch 700, Loss: 4646.0801\
Epoch 800, Loss: 4646.0767\
Epoch 900, Loss: 4646.0747\
Epoch 1000, Loss: 4646.0723\
Epoch 1100, Loss: 4646.0688\
Epoch 1200, Loss: 4646.0664\
Epoch 1300, Loss: 4646.0630\
Epoch 1400, Loss: 4646.0601\
Epoch 1500, Loss: 4646.0571\
Epoch 1600, Loss: 4646.0542\
Epoch 1700, Loss: 4646.0522\
Epoch 1800, Loss: 4646.0488\
Epoch 1900, Loss: 4646.0469\
Epoch 2000, Loss: 4646.0435\
Epoch 2100, Loss: 4646.0410\
Epoch 2200, Loss: 4646.0386\
Epoch 2300, Loss: 4646.0356\
Epoch 2400, Loss: 4646.0342\
Epoch 2500, Loss: 4646.0303\
Epoch 2600, Loss: 4646.0283\
Epoch 2700, Loss: 4646.0259\
Epoch 2800, Loss: 4646.0229\
Epoch 2900, Loss: 4646.0210\
Epoch 3000, Loss: 4646.0190\
Epoch 3100, Loss: 4646.0161\
Epoch 3200, Loss: 4646.0137\
Epoch 3300, Loss: 4646.0122\
Epoch 3400, Loss: 4646.0093\
Epoch 3500, Loss: 4646.0078\
Epoch 3600, Loss: 4646.0054\
Epoch 3700, Loss: 4646.0024\
Epoch 3800, Loss: 4646.0010\
Epoch 3900, Loss: 4645.9985\
Epoch 4000, Loss: 4645.9966\
Epoch 4100, Loss: 4645.9941\
Epoch 4200, Loss: 4645.9927\
Epoch 4300, Loss: 4645.9902\
Epoch 4400, Loss: 4645.9888\
Epoch 4500, Loss: 4645.9868\
Epoch 4600, Loss: 4645.9849\
Epoch 4700, Loss: 4645.9819\
Epoch 4800, Loss: 4645.9814\
Epoch 4900, Loss: 4645.9790\
Epoch 5000, Loss: 4645.9785\
Epoch 5100, Loss: 4645.9761\
Epoch 5200, Loss: 4645.9746\
Epoch 5300, Loss: 4645.9727\
Epoch 5400, Loss: 4645.9722\
Epoch 5500, Loss: 4645.9697\
Epoch 5600, Loss: 4645.9678\
Epoch 5700, Loss: 4645.9668\
Epoch 5800, Loss: 4645.9648\
Epoch 5900, Loss: 4645.9639\
Epoch 6000, Loss: 4645.9629\
Epoch 6100, Loss: 4645.9614\
Epoch 6200, Loss: 4645.9604\
Epoch 6300, Loss: 4645.9590\
Epoch 6400, Loss: 4645.9575\
Epoch 6500, Loss: 4645.9556\
Epoch 6600, Loss: 4645.9546\
Epoch 6700, Loss: 4645.9541\
Epoch 6800, Loss: 4645.9531\
Epoch 6900, Loss: 4645.9517\
Epoch 7000, Loss: 4645.9512\
Epoch 7100, Loss: 4645.9502\
Epoch 7200, Loss: 4645.9487\
Epoch 7300, Loss: 4645.9487\
Epoch 7400, Loss: 4645.9473\
Epoch 7500, Loss: 4645.9463\
Epoch 7600, Loss: 4645.9473\
Epoch 7700, Loss: 4645.9453\
Epoch 7800, Loss: 4645.9443\
Epoch 7900, Loss: 4645.9443\
Epoch 8000, Loss: 4645.9438\
Epoch 8100, Loss: 4645.9434\
Epoch 8200, Loss: 4645.9424\
Epoch 8300, Loss: 4645.9419\
Epoch 8400, Loss: 4645.9414\
Epoch 8500, Loss: 4645.9419\
Epoch 8600, Loss: 4645.9414\
Epoch 8700, Loss: 4645.9409\
Epoch 8800, Loss: 4645.9399\
Epoch 8900, Loss: 4645.9409\
Epoch 9000, Loss: 4645.9404\
Epoch 9100, Loss: 4645.9409\
Epoch 9200, Loss: 4645.9414\
Epoch 9300, Loss: 4645.9409\
Epoch 9400, Loss: 4645.9414\
Epoch 9500, Loss: 4645.9414\
Epoch 9600, Loss: 4645.9419\
Epoch 9700, Loss: 4645.9409\
Epoch 9800, Loss: 4645.9419\
Epoch 9900, Loss: 4645.9414\
Epoch 10000, Loss: 4645.9414\
target k_spring: [2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]\
Optimized k_spring: [  0.92316777   2.6535547    3.518252     4.5302386    3.9465086\
   1.2087649    1.9824193    1.620018     3.0672805    2.291326\
   5.275871     4.6516004    4.4615884    2.4750798    0.32148167\
   4.0870843  -12.301869     3.2820954    1.0630897    4.6811543 ]\
Final Loss: 4645.94140625\
\
# Simple GPU final frame
498MiB gpu usage: faster but low precision(f64 is not supported)

Initial k\_spring:\[1.5832518 1.721834  2.057561  2.0346317 1.8523902 1.874951  2.1294842  
1.553362  2.1117568 1.6693558 2.0058918 2.2758453 2.0924106 1.7392561  
1.9993211 1.9175192 2.237509  2.1976027 1.6197121 1.6779879\]  

Epoch 100, Loss: 4613.4707\  
Epoch 200, Loss: 4613.3999\  
Epoch 300, Loss: 4613.3281\  
Epoch 400, Loss: 4613.2559\  
Epoch 500, Loss: 4613.1826\  
Epoch 600, Loss: 4613.1084\  
Epoch 700, Loss: 4613.0337\  
Epoch 800, Loss: 4612.9585\  
Epoch 900, Loss: 4612.8823\  
Epoch 1000, Loss: 4612.8052\  
Epoch 1100, Loss: 4612.7271\  
Epoch 1200, Loss: 4612.6484\  
Epoch 1300, Loss: 4612.5693\  
Epoch 1400, Loss: 4612.4893\  
Epoch 1500, Loss: 4612.4082\  
Epoch 1600, Loss: 4612.3262\  
Epoch 1700, Loss: 4612.2437\  
Epoch 1800, Loss: 4612.1597\  
Epoch 1900, Loss: 4612.0752\  
Epoch 2000, Loss: 4611.9897\  
Epoch 2100, Loss: 4611.9033\  
Epoch 2200, Loss: 4611.8159\  
Epoch 2300, Loss: 4611.7275\  
Epoch 2400, Loss: 4611.6382\  
Epoch 2500, Loss: 4611.5479\  
Epoch 2600, Loss: 4611.4570\  
Epoch 2700, Loss: 4611.3647\  
Epoch 2800, Loss: 4611.2710\  
Epoch 2900, Loss: 4611.1768\  
Epoch 3000, Loss: 4611.0815\  
Epoch 3100, Loss: 4610.9849\  
Epoch 3200, Loss: 4610.8877\  
Epoch 3300, Loss: 4610.7886\  
Epoch 3400, Loss: 4610.6890\  
Epoch 3500, Loss: 4610.5879\  
Epoch 3600, Loss: 4610.4858\  
Epoch 3700, Loss: 4610.3828\  
Epoch 3800, Loss: 4610.2783\  
Epoch 3900, Loss: 4610.1729\  
Epoch 4000, Loss: 4610.0659\  
Epoch 4100, Loss: 4609.9575\  
Epoch 4200, Loss: 4609.8486\  
Epoch 4300, Loss: 4609.7373\  
Epoch 4400, Loss: 4609.6260\  
Epoch 4500, Loss: 4609.5127\  
Epoch 4600, Loss: 4609.3984\  
Epoch 4700, Loss: 4609.2822\  
Epoch 4800, Loss: 4609.1650\  
Epoch 4900, Loss: 4609.0469\  
Epoch 5000, Loss: 4608.9272\  
Epoch 5100, Loss: 4608.8057\  
Epoch 5200, Loss: 4608.6831\  
Epoch 5300, Loss: 4608.5591\  
Epoch 5400, Loss: 4608.4336\  
Epoch 5500, Loss: 4608.3066\  
Epoch 5600, Loss: 4608.1787\  
Epoch 5700, Loss: 4608.0488\  
Epoch 5800, Loss: 4607.9175\  
Epoch 5900, Loss: 4607.7852\  
Epoch 6000, Loss: 4607.6509\  
Epoch 6100, Loss: 4607.5156\  
Epoch 6200, Loss: 4607.3784\  
Epoch 6300, Loss: 4607.2402\  
Epoch 6400, Loss: 4607.1006\  
Epoch 6500, Loss: 4606.9590\  
Epoch 6600, Loss: 4606.8169\  
Epoch 6700, Loss: 4606.6729\  
Epoch 6800, Loss: 4606.5278\  
Epoch 6900, Loss: 4606.3809\  
Epoch 7000, Loss: 4606.2329\  
Epoch 7100, Loss: 4606.0840\  
Epoch 7200, Loss: 4605.9336\  
Epoch 7300, Loss: 4605.7822\  
Epoch 7400, Loss: 4605.6299\  
Epoch 7500, Loss: 4605.4766\  
Epoch 7600, Loss: 4605.3218\  
Epoch 7700, Loss: 4605.1670\  
Epoch 7800, Loss: 4605.0107\  
Epoch 7900, Loss: 4604.8545\  
Epoch 8000, Loss: 4604.6973\  
Epoch 8100, Loss: 4604.5405\  
Epoch 8200, Loss: 4604.3828\  
Epoch 8300, Loss: 4604.2256\  
Epoch 8400, Loss: 4604.0688\  
Epoch 8500, Loss: 4603.9126\  
Epoch 8600, Loss: 4603.7568\  
Epoch 8700, Loss: 4603.6025\  
Epoch 8800, Loss: 4603.4492\  
Epoch 8900, Loss: 4603.2983\  
Epoch 9000, Loss: 4603.1494\  
Epoch 9100, Loss: 4603.0029\  
Epoch 9200, Loss: 4602.8599\  
Epoch 9300, Loss: 4602.7212\  
Epoch 9400, Loss: 4602.5864\  
Epoch 9500, Loss: 4602.4570\  
Epoch 9600, Loss: 4602.3340\  
Epoch 9700, Loss: 4602.2173\  
Epoch 9800, Loss: 4602.1089\  
Epoch 9900, Loss: 4602.0098\  
Epoch 10000, Loss: 4601.9209\

target k\_spring:\[2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.\]  
Optimized k\_spring:\[  1.6125094   2.3781924   2.1874704   2.527857    2.987854    1.9833714  
   3.3573203   1.553362    2.1236777   3.2248359   2.668384    2.2758453  
   3.4571137 -42.882263    2.131616  -19.983997    3.1406524   2.2190604  
   1.7876874   1.9903654\]  
Final Loss: 4601.9208984375

# simple gpu single frame
\[Taichi\] version 1.7.2, llvm 15.0.4, commit 0131dce9, linux, python 3.10.12  
\[Taichi\] Starting on arch=cuda  
Initial k\_spring:\[1.6377798 1.7041128 1.8542019 2.2286425 1.8974178 1.7165663 1.9187601  
2.2324238 1.8327324 1.7337582 1.8539207 2.267703  1.7678223 2.0159423  
1.7936084 1.5577222 1.9147947 1.9399893 1.8480138 1.520158\]  
Epoch 100, Loss: 4552.8867\  
Epoch 200, Loss: 4551.8843\  
Epoch 300, Loss: 4550.8203\  
Epoch 400, Loss: 4549.6997\  
Epoch 500, Loss: 4548.5356\  
Epoch 600, Loss: 4547.3506\  
Epoch 700, Loss: 4546.1860\  
Epoch 800, Loss: 4545.1108\  
Epoch 900, Loss: 4544.2363\  
Epoch 1000, Loss: 4543.7329\  
target k\_spring:\[2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.\]  
Optimized k\_spring:\[  1.6559323 -16.478842    2.3177729   2.2570143 -16.178165    2.0866604  
   2.004702  -29.604483    2.852794    1.9722556  -4.508065    2.4128313  
   1.9426762   2.0199955   1.8627756   1.5577222   3.0435472   2.1543217  
   3.4273      1.915492\]  
Final Loss: 4543.73291015625  

# cpu multi-frames
![image2](/2.jpg)
[Taichi] version 1.7.2, llvm 15.0.4, commit 0131dce9, linux, python 3.10.12\
[Taichi] Starting on arch=x64\
Initial k_spring: [2.102127  2.213722  2.0862687 1.9338472 1.5172333 2.2400713 1.8320391
 1.6787615 1.7252984 1.9037942 2.2860355 1.933124  1.9878671 2.0032115
 1.5248609 1.5886344 2.283393  1.9411702 1.6583196 2.003668 ]\
Epoch 100, Loss: 4490.4595\
Epoch 200, Loss: 4487.5649\
Epoch 300, Loss: 4484.3901\
Epoch 400, Loss: 4480.9092\
Epoch 500, Loss: 4477.1006\
Epoch 600, Loss: 4472.9658\
Epoch 700, Loss: 4468.5459\
Epoch 800, Loss: 4463.9800\
Epoch 900, Loss: 4459.6211\
Epoch 1000, Loss: 4456.3052\
target k_spring: [2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]\
Optimized k_spring: [  2.102127    2.6393273   2.0862687   2.3735476   1.6083443 -41.723858
   2.4206564   2.412927    1.9159497   3.6071384   2.3386858 -21.866106
 -21.952497  -31.0159      1.5451264   1.5944756   3.1373527 -15.7339325
   2.2233145 -23.900965 ]\
Final Loss: 4456.30517578125

# Overall
It could learn something, but the grad has issues and cannot converge