#Implementation of the feedforward neural network with cifar10 dataset

#first decompression the dataset 'cifar-10-binary.tar.gz' to the fold './data/'

Run the shell code below:  
```shell
tar -xzvf cifar-10-binary.tar.gz
```

Then will appear a fold 'cifar-10-batches-bin'.  
And the files in the fold:  
```shell
batches.meta.txt
data_batch_1.bin
data_batch_2.bin
data_batch_3.bin
data_batch_4.bin
data_batch_5.bin
readme.html
test_batch.bin
```

Run the shell bellow:  
```shell
bash run.sh
```




Then output
```shell
pyperparameters:
learn_rate:0.005
epsilon:1e-07
output_dim:10
batch_size:128
file_path:./data/
reg_lambda:0.05
activation:sigmoid
hidden_dim:2048
input_dim:3072
max_epochs:50
max_steps:50000
train_steps:50
architecture:[3072, 2048, 10]
E:0 S:50 Train_Loss:2.40109 Test_Loss:2.36403 Train_Acc:0.08594 Test_Acc:0.11798 gap:-0.03204 Train_Speed:1196.95652911 Test_Speed:4396.79711724
best_epoch:0 best_test_acc:0.117978639241
E:0 S:100 Train_Loss:2.73293 Test_Loss:2.86622 Train_Acc:0.12500 Test_Acc:0.14735 gap:-0.02235 Train_Speed:1186.44981017 Test_Speed:3955.98670705
best_epoch:0 best_test_acc:0.147349683544
E:0 S:150 Train_Loss:3.19594 Test_Loss:3.08659 Train_Acc:0.21875 Test_Acc:0.19027 gap:0.02848 Train_Speed:1139.08183629 Test_Speed:4421.85690165
best_epoch:0 best_test_acc:0.190268987342
E:0 S:200 Train_Loss:3.84240 Test_Loss:3.45429 Train_Acc:0.17969 Test_Acc:0.20144 gap:-0.02176 Train_Speed:1020.29470611 Test_Speed:2919.56969226
best_epoch:0 best_test_acc:0.201443829114
E:0 S:250 Train_Loss:2.59963 Test_Loss:3.05986 Train_Acc:0.21094 Test_Acc:0.19759 gap:0.01335 Train_Speed:1047.98650759 Test_Speed:3492.77473668
best_epoch:0 best_test_acc:0.201443829114
E:0 S:300 Train_Loss:3.47834 Test_Loss:3.65382 Train_Acc:0.14844 Test_Acc:0.20688 gap:-0.05845 Train_Speed:957.839422518 Test_Speed:3265.30049812
best_epoch:0 best_test_acc:0.206882911392
E:0 S:350 Train_Loss:2.79823 Test_Loss:3.33488 Train_Acc:0.24219 Test_Acc:0.22063 gap:0.02156 Train_Speed:1107.5829123 Test_Speed:4329.32481775
best_epoch:0 best_test_acc:0.220628955696
E:1 S:400 Train_Loss:3.29347 Test_Loss:3.32084 Train_Acc:0.24219 Test_Acc:0.23022 gap:0.01197 Train_Speed:1022.81387551 Test_Speed:3619.8262605
best_epoch:1 best_test_acc:0.230221518987
E:1 S:450 Train_Loss:3.66320 Test_Loss:3.18382 Train_Acc:0.17188 Test_Acc:0.22083 gap:-0.04895 Train_Speed:1033.77604221 Test_Speed:3801.50193307
best_epoch:1 best_test_acc:0.230221518987
E:1 S:500 Train_Loss:3.46018 Test_Loss:3.05441 Train_Acc:0.17969 Test_Acc:0.23200 gap:-0.05231 Train_Speed:1054.55731728 Test_Speed:2861.9988272
best_epoch:1 best_test_acc:0.232001582278
E:1 S:550 Train_Loss:3.19501 Test_Loss:3.25045 Train_Acc:0.21094 Test_Acc:0.22884 gap:-0.01790 Train_Speed:1078.60200463 Test_Speed:2156.15137653
best_epoch:1 best_test_acc:0.232001582278
E:1 S:600 Train_Loss:3.27641 Test_Loss:3.39094 Train_Acc:0.25781 Test_Acc:0.22785 gap:0.02996 Train_Speed:1104.35453162 Test_Speed:1596.927071
best_epoch:1 best_test_acc:0.232001582278
E:1 S:650 Train_Loss:3.69020 Test_Loss:3.44635 Train_Acc:0.30469 Test_Acc:0.23299 gap:0.07170 Train_Speed:1148.32311358 Test_Speed:4115.0885455
best_epoch:1 best_test_acc:0.232990506329
E:1 S:700 Train_Loss:3.41978 Test_Loss:3.65220 Train_Acc:0.21094 Test_Acc:0.22439 gap:-0.01345 Train_Speed:1127.96300568 Test_Speed:4346.32345393
best_epoch:1 best_test_acc:0.232990506329
E:1 S:750 Train_Loss:4.83669 Test_Loss:3.69271 Train_Acc:0.17188 Test_Acc:0.22725 gap:-0.05538 Train_Speed:1152.00192261 Test_Speed:4489.18750418
best_epoch:1 best_test_acc:0.232990506329
E:2 S:800 Train_Loss:4.27469 Test_Loss:3.61413 Train_Acc:0.28125 Test_Acc:0.22112 gap:0.06013 Train_Speed:1220.90758214 Test_Speed:4418.07246723
best_epoch:1 best_test_acc:0.232990506329
E:2 S:850 Train_Loss:3.60413 Test_Loss:3.57808 Train_Acc:0.23438 Test_Acc:0.23329 gap:0.00109 Train_Speed:1153.19462744 Test_Speed:4376.51043849
best_epoch:2 best_test_acc:0.233287183544
E:2 S:900 Train_Loss:4.30450 Test_Loss:4.45190 Train_Acc:0.16406 Test_Acc:0.19482 gap:-0.03076 Train_Speed:1208.02599343 Test_Speed:4029.47335555
best_epoch:2 best_test_acc:0.233287183544
E:2 S:950 Train_Loss:3.46003 Test_Loss:3.87522 Train_Acc:0.23438 Test_Acc:0.22627 gap:0.00811 Train_Speed:1052.51883423 Test_Speed:4551.45064261
best_epoch:2 best_test_acc:0.233287183544
E:2 S:1000 Train_Loss:4.13542 Test_Loss:3.67394 Train_Acc:0.15625 Test_Acc:0.22627 gap:-0.07002 Train_Speed:1064.42174718 Test_Speed:4086.7086245
best_epoch:2 best_test_acc:0.233287183544
E:2 S:1050 Train_Loss:4.47440 Test_Loss:3.42450 Train_Acc:0.20312 Test_Acc:0.23240 gap:-0.02927 Train_Speed:1202.16152696 Test_Speed:1849.06650318
best_epoch:2 best_test_acc:0.233287183544
E:2 S:1100 Train_Loss:3.42983 Test_Loss:3.46037 Train_Acc:0.24219 Test_Acc:0.23378 gap:0.00841 Train_Speed:1192.55089996 Test_Speed:4545.28524501
best_epoch:2 best_test_acc:0.23378164557
E:2 S:1150 Train_Loss:3.74305 Test_Loss:4.28859 Train_Acc:0.27344 Test_Acc:0.22824 gap:0.04519 Train_Speed:1212.62634449 Test_Speed:4841.90937951
best_epoch:2 best_test_acc:0.23378164557
E:3 S:1200 Train_Loss:3.76206 Test_Loss:3.55129 Train_Acc:0.19531 Test_Acc:0.22310 gap:-0.02779 Train_Speed:1127.22646533 Test_Speed:4313.81001816
best_epoch:2 best_test_acc:0.23378164557
E:3 S:1250 Train_Loss:4.28606 Test_Loss:3.66876 Train_Acc:0.20312 Test_Acc:0.22112 gap:-0.01800 Train_Speed:729.008353702 Test_Speed:4447.38818384
best_epoch:2 best_test_acc:0.23378164557
E:3 S:1300 Train_Loss:3.53226 Test_Loss:3.73910 Train_Acc:0.28906 Test_Acc:0.22597 gap:0.06309 Train_Speed:1211.21469148 Test_Speed:4358.03680464
best_epoch:2 best_test_acc:0.23378164557
E:3 S:1350 Train_Loss:5.38663 Test_Loss:4.59524 Train_Acc:0.17969 Test_Acc:0.18018 gap:-0.00049 Train_Speed:1144.28712527 Test_Speed:4323.43277741
best_epoch:2 best_test_acc:0.23378164557
E:3 S:1400 Train_Loss:2.33098 Test_Loss:3.50315 Train_Acc:0.33594 Test_Acc:0.23883 gap:0.09711 Train_Speed:1176.40427201 Test_Speed:4242.49610418
best_epoch:3 best_test_acc:0.238825158228
E:3 S:1450 Train_Loss:3.68373 Test_Loss:3.56768 Train_Acc:0.25781 Test_Acc:0.23438 gap:0.02344 Train_Speed:1181.28091322 Test_Speed:4338.52609802
best_epoch:3 best_test_acc:0.238825158228
E:3 S:1500 Train_Loss:3.55359 Test_Loss:3.43764 Train_Acc:0.27344 Test_Acc:0.23803 gap:0.03540 Train_Speed:1195.1898449 Test_Speed:4333.3783618
best_epoch:3 best_test_acc:0.238825158228
E:3 S:1550 Train_Loss:3.81624 Test_Loss:3.67806 Train_Acc:0.21094 Test_Acc:0.22834 gap:-0.01741 Train_Speed:1177.21682882 Test_Speed:3657.78172032
best_epoch:3 best_test_acc:0.238825158228
E:4 S:1600 Train_Loss:4.00349 Test_Loss:3.45820 Train_Acc:0.19531 Test_Acc:0.22716 gap:-0.03184 Train_Speed:1172.09718738 Test_Speed:2576.48981394
best_epoch:3 best_test_acc:0.238825158228
E:4 S:1650 Train_Loss:3.18337 Test_Loss:3.47405 Train_Acc:0.28906 Test_Acc:0.24080 gap:0.04826 Train_Speed:1212.5578746 Test_Speed:4253.04924266
best_epoch:4 best_test_acc:0.240803006329
E:4 S:1700 Train_Loss:3.31928 Test_Loss:3.50022 Train_Acc:0.25000 Test_Acc:0.22814 gap:0.02186 Train_Speed:1132.56259493 Test_Speed:4611.6195401
best_epoch:4 best_test_acc:0.240803006329
E:4 S:1750 Train_Loss:2.78207 Test_Loss:3.33416 Train_Acc:0.32031 Test_Acc:0.24021 gap:0.08010 Train_Speed:1101.30303372 Test_Speed:3714.98399474
best_epoch:4 best_test_acc:0.240803006329
E:4 S:1800 Train_Loss:3.21437 Test_Loss:3.06041 Train_Acc:0.23438 Test_Acc:0.25524 gap:-0.02087 Train_Speed:1123.88272803 Test_Speed:2048.52357132
best_epoch:4 best_test_acc:0.255241297468
E:4 S:1850 Train_Loss:2.76578 Test_Loss:3.30497 Train_Acc:0.28125 Test_Acc:0.23012 gap:0.05113 Train_Speed:1155.85118013 Test_Speed:4870.46096344
best_epoch:4 best_test_acc:0.255241297468
E:4 S:1900 Train_Loss:4.05632 Test_Loss:3.19663 Train_Acc:0.19531 Test_Acc:0.24397 gap:-0.04866 Train_Speed:1083.59823353 Test_Speed:3997.13292732
best_epoch:4 best_test_acc:0.255241297468
E:4 S:1950 Train_Loss:2.91405 Test_Loss:3.38560 Train_Acc:0.21875 Test_Acc:0.23230 gap:-0.01355 Train_Speed:1171.46547383 Test_Speed:4092.19104532
best_epoch:4 best_test_acc:0.255241297468
E:5 S:2000 Train_Loss:3.53234 Test_Loss:3.26789 Train_Acc:0.18750 Test_Acc:0.23586 gap:-0.04836 Train_Speed:1186.14049755 Test_Speed:4045.50525967
best_epoch:4 best_test_acc:0.255241297468
E:5 S:2050 Train_Loss:2.98674 Test_Loss:2.92163 Train_Acc:0.27344 Test_Acc:0.25475 gap:0.01869 Train_Speed:1127.84215657 Test_Speed:4047.30463102
best_epoch:4 best_test_acc:0.255241297468
E:5 S:2100 Train_Loss:3.03509 Test_Loss:3.14473 Train_Acc:0.24219 Test_Acc:0.24031 gap:0.00188 Train_Speed:1121.19529361 Test_Speed:4454.65787137
best_epoch:4 best_test_acc:0.255241297468
E:5 S:2150 Train_Loss:3.29612 Test_Loss:3.11643 Train_Acc:0.23438 Test_Acc:0.23507 gap:-0.00069 Train_Speed:1052.83050452 Test_Speed:4373.05252183
best_epoch:4 best_test_acc:0.255241297468
E:5 S:2200 Train_Loss:3.33340 Test_Loss:2.87789 Train_Acc:0.18750 Test_Acc:0.25326 gap:-0.06576 Train_Speed:1209.29128403 Test_Speed:4496.10505159
best_epoch:4 best_test_acc:0.255241297468
E:5 S:2250 Train_Loss:2.99067 Test_Loss:2.95525 Train_Acc:0.20312 Test_Acc:0.24644 gap:-0.04331 Train_Speed:1172.8576809 Test_Speed:4567.67581272
best_epoch:4 best_test_acc:0.255241297468
E:5 S:2300 Train_Loss:2.87697 Test_Loss:3.11992 Train_Acc:0.30469 Test_Acc:0.24080 gap:0.06388 Train_Speed:1168.53324047 Test_Speed:4591.269462
best_epoch:4 best_test_acc:0.255241297468
E:6 S:2350 Train_Loss:3.13953 Test_Loss:3.04756 Train_Acc:0.21875 Test_Acc:0.24802 gap:-0.02927 Train_Speed:1207.44457689 Test_Speed:4524.10412155
best_epoch:4 best_test_acc:0.255241297468
E:6 S:2400 Train_Loss:2.98216 Test_Loss:3.08932 Train_Acc:0.25781 Test_Acc:0.22261 gap:0.03521 Train_Speed:1200.55930445 Test_Speed:4667.95562202
best_epoch:4 best_test_acc:0.255241297468
E:6 S:2450 Train_Loss:2.64439 Test_Loss:2.69463 Train_Acc:0.20312 Test_Acc:0.26147 gap:-0.05835 Train_Speed:1114.4111443 Test_Speed:4953.36911934
best_epoch:6 best_test_acc:0.261471518987
E:6 S:2500 Train_Loss:2.96564 Test_Loss:2.77728 Train_Acc:0.26562 Test_Acc:0.25415 gap:0.01147 Train_Speed:1167.10560389 Test_Speed:4997.86736176
best_epoch:6 best_test_acc:0.261471518987
E:6 S:2550 Train_Loss:2.79762 Test_Loss:2.60256 Train_Acc:0.27344 Test_Acc:0.26948 gap:0.00396 Train_Speed:1188.54321294 Test_Speed:4400.29270212
best_epoch:6 best_test_acc:0.269481803797
E:6 S:2600 Train_Loss:3.05216 Test_Loss:2.65896 Train_Acc:0.27344 Test_Acc:0.24219 gap:0.03125 Train_Speed:1179.79864323 Test_Speed:3713.36518696
best_epoch:6 best_test_acc:0.269481803797
E:6 S:2650 Train_Loss:2.64152 Test_Loss:2.74660 Train_Acc:0.29688 Test_Acc:0.24268 gap:0.05419 Train_Speed:1183.3456663 Test_Speed:4086.98861915
best_epoch:6 best_test_acc:0.269481803797
E:6 S:2700 Train_Loss:2.67641 Test_Loss:2.69596 Train_Acc:0.32031 Test_Acc:0.25752 gap:0.06280 Train_Speed:1191.00852319 Test_Speed:4357.89530419
best_epoch:6 best_test_acc:0.269481803797
E:7 S:2750 Train_Loss:3.38848 Test_Loss:3.04510 Train_Acc:0.21875 Test_Acc:0.23081 gap:-0.01206 Train_Speed:1178.92038507 Test_Speed:4261.82732671
best_epoch:6 best_test_acc:0.269481803797
E:7 S:2800 Train_Loss:2.92383 Test_Loss:2.62347 Train_Acc:0.30469 Test_Acc:0.25069 gap:0.05400 Train_Speed:1024.04883045 Test_Speed:2370.3642585
best_epoch:6 best_test_acc:0.269481803797
E:7 S:2850 Train_Loss:2.77859 Test_Loss:2.46160 Train_Acc:0.25781 Test_Acc:0.27611 gap:-0.01830 Train_Speed:1216.7983754 Test_Speed:3980.09409218
best_epoch:7 best_test_acc:0.276107594937
E:7 S:2900 Train_Loss:3.13493 Test_Loss:2.72927 Train_Acc:0.18750 Test_Acc:0.26315 gap:-0.07565 Train_Speed:1172.26866037 Test_Speed:3305.10235968
best_epoch:7 best_test_acc:0.276107594937
E:7 S:2950 Train_Loss:2.52136 Test_Loss:2.53657 Train_Acc:0.22656 Test_Acc:0.26909 gap:-0.04252 Train_Speed:1078.20342979 Test_Speed:3911.25730896
best_epoch:7 best_test_acc:0.276107594937
E:7 S:3000 Train_Loss:2.28862 Test_Loss:2.50843 Train_Acc:0.31250 Test_Acc:0.24911 gap:0.06339 Train_Speed:1191.64033552 Test_Speed:4173.30704891
best_epoch:7 best_test_acc:0.276107594937
E:7 S:3050 Train_Loss:2.69223 Test_Loss:2.59592 Train_Acc:0.26562 Test_Acc:0.25613 gap:0.00949 Train_Speed:1166.09668115 Test_Speed:4135.18379419
best_epoch:7 best_test_acc:0.276107594937
E:7 S:3100 Train_Loss:2.40791 Test_Loss:2.45942 Train_Acc:0.27344 Test_Acc:0.27225 gap:0.00119 Train_Speed:1125.90736399 Test_Speed:2323.13093147
best_epoch:7 best_test_acc:0.276107594937
E:8 S:3150 Train_Loss:2.34843 Test_Loss:2.48130 Train_Acc:0.28125 Test_Acc:0.26849 gap:0.01276 Train_Speed:1194.38683856 Test_Speed:3490.02738087
best_epoch:7 best_test_acc:0.276107594937
E:8 S:3200 Train_Loss:2.43000 Test_Loss:2.42790 Train_Acc:0.25781 Test_Acc:0.26404 gap:-0.00623 Train_Speed:1221.80504679 Test_Speed:4578.15356278
best_epoch:7 best_test_acc:0.276107594937
E:8 S:3250 Train_Loss:2.51207 Test_Loss:2.39278 Train_Acc:0.22656 Test_Acc:0.27611 gap:-0.04955 Train_Speed:1215.40176083 Test_Speed:4282.53082648
best_epoch:7 best_test_acc:0.276107594937
E:8 S:3300 Train_Loss:2.28570 Test_Loss:2.33994 Train_Acc:0.32031 Test_Acc:0.28738 gap:0.03293 Train_Speed:1221.17696084 Test_Speed:4558.25192732
best_epoch:8 best_test_acc:0.287381329114
E:8 S:3350 Train_Loss:2.16668 Test_Loss:2.37339 Train_Acc:0.31250 Test_Acc:0.27858 gap:0.03392 Train_Speed:1119.23421803 Test_Speed:4051.91710064
best_epoch:8 best_test_acc:0.287381329114
E:8 S:3400 Train_Loss:2.33357 Test_Loss:2.39103 Train_Acc:0.32031 Test_Acc:0.27739 gap:0.04292 Train_Speed:1179.14565401 Test_Speed:4524.10412155
best_epoch:8 best_test_acc:0.287381329114
E:8 S:3450 Train_Loss:2.57082 Test_Loss:2.32535 Train_Acc:0.21875 Test_Acc:0.28135 gap:-0.06260 Train_Speed:1147.57201731 Test_Speed:4240.38505951
best_epoch:8 best_test_acc:0.287381329114
E:8 S:3500 Train_Loss:2.48471 Test_Loss:2.42561 Train_Acc:0.30469 Test_Acc:0.27413 gap:0.03056 Train_Speed:1167.33907069 Test_Speed:4199.7505515
best_epoch:8 best_test_acc:0.287381329114
E:9 S:3550 Train_Loss:2.42667 Test_Loss:2.41197 Train_Acc:0.29688 Test_Acc:0.27769 gap:0.01919 Train_Speed:1165.60805744 Test_Speed:4193.6815004
best_epoch:8 best_test_acc:0.287381329114
E:9 S:3600 Train_Loss:2.51808 Test_Loss:2.34828 Train_Acc:0.22656 Test_Acc:0.27383 gap:-0.04727 Train_Speed:916.42625945 Test_Speed:4169.51493076
best_epoch:8 best_test_acc:0.287381329114
E:9 S:3650 Train_Loss:2.47517 Test_Loss:2.29117 Train_Acc:0.21875 Test_Acc:0.28471 gap:-0.06596 Train_Speed:894.810206289 Test_Speed:2389.84233931
best_epoch:8 best_test_acc:0.287381329114
E:9 S:3700 Train_Loss:2.30268 Test_Loss:2.28041 Train_Acc:0.28125 Test_Acc:0.27838 gap:0.00287 Train_Speed:1198.9051208 Test_Speed:4522.65588381
best_epoch:8 best_test_acc:0.287381329114
E:9 S:3750 Train_Loss:2.17532 Test_Loss:2.27726 Train_Acc:0.31250 Test_Acc:0.29678 gap:0.01572 Train_Speed:1188.26699719 Test_Speed:4831.27777978
best_epoch:9 best_test_acc:0.296776107595
E:9 S:3800 Train_Loss:2.46029 Test_Loss:2.26493 Train_Acc:0.31250 Test_Acc:0.28135 gap:0.03115 Train_Speed:1273.45526654 Test_Speed:4857.90084604
best_epoch:9 best_test_acc:0.296776107595
E:9 S:3850 Train_Loss:2.04513 Test_Loss:2.38796 Train_Acc:0.32031 Test_Acc:0.26701 gap:0.05330 Train_Speed:1103.55321043 Test_Speed:4519.15346089
best_epoch:9 best_test_acc:0.296776107595
E:9 S:3900 Train_Loss:2.66478 Test_Loss:2.31545 Train_Acc:0.23438 Test_Acc:0.28610 gap:-0.05172 Train_Speed:1060.8943929 Test_Speed:4518.46883863
best_epoch:9 best_test_acc:0.296776107595
E:10 S:3950 Train_Loss:2.51084 Test_Loss:2.40000 Train_Acc:0.30469 Test_Acc:0.27077 gap:0.03392 Train_Speed:1210.99612478 Test_Speed:4583.70397691
best_epoch:9 best_test_acc:0.296776107595
E:10 S:4000 Train_Loss:2.33085 Test_Loss:2.30818 Train_Acc:0.27344 Test_Acc:0.28333 gap:-0.00989 Train_Speed:1091.57377316 Test_Speed:3022.49633779
best_epoch:9 best_test_acc:0.296776107595
E:10 S:4050 Train_Loss:2.81832 Test_Loss:2.24091 Train_Acc:0.22656 Test_Acc:0.28768 gap:-0.06112 Train_Speed:1113.64367694 Test_Speed:4559.37454459
best_epoch:9 best_test_acc:0.296776107595
E:10 S:4100 Train_Loss:2.28684 Test_Loss:2.44056 Train_Acc:0.29688 Test_Acc:0.27205 gap:0.02482 Train_Speed:749.814473363 Test_Speed:4034.80318653
best_epoch:9 best_test_acc:0.296776107595
E:10 S:4150 Train_Loss:2.43879 Test_Loss:2.24455 Train_Acc:0.27344 Test_Acc:0.28026 gap:-0.00682 Train_Speed:1191.75143455 Test_Speed:4189.85228195
best_epoch:9 best_test_acc:0.296776107595
E:10 S:4200 Train_Loss:2.37320 Test_Loss:2.17202 Train_Acc:0.32031 Test_Acc:0.29945 gap:0.02087 Train_Speed:1156.89479423 Test_Speed:3642.27213026
best_epoch:10 best_test_acc:0.299446202532
E:10 S:4250 Train_Loss:2.32131 Test_Loss:2.30443 Train_Acc:0.35156 Test_Acc:0.29173 gap:0.05983 Train_Speed:676.393254095 Test_Speed:4281.06240531
best_epoch:10 best_test_acc:0.299446202532
E:10 S:4300 Train_Loss:2.44296 Test_Loss:2.24309 Train_Acc:0.28906 Test_Acc:0.29173 gap:-0.00267 Train_Speed:1041.9619835 Test_Speed:4106.77828774
best_epoch:10 best_test_acc:0.299446202532
E:11 S:4350 Train_Loss:2.15289 Test_Loss:2.24860 Train_Acc:0.27344 Test_Acc:0.27828 gap:-0.00485 Train_Speed:1090.70889152 Test_Speed:4325.4879389
best_epoch:10 best_test_acc:0.299446202532
E:11 S:4400 Train_Loss:2.66482 Test_Loss:2.22693 Train_Acc:0.25000 Test_Acc:0.28214 gap:-0.03214 Train_Speed:1193.13927083 Test_Speed:4414.69379163
best_epoch:10 best_test_acc:0.299446202532
E:11 S:4450 Train_Loss:2.36792 Test_Loss:2.22178 Train_Acc:0.30469 Test_Acc:0.28165 gap:0.02304 Train_Speed:722.323011183 Test_Speed:4381.76122229
best_epoch:10 best_test_acc:0.299446202532
E:11 S:4500 Train_Loss:2.14378 Test_Loss:2.21626 Train_Acc:0.28125 Test_Acc:0.27393 gap:0.00732 Train_Speed:1052.85734569 Test_Speed:4148.73276355
best_epoch:10 best_test_acc:0.299446202532
E:11 S:4550 Train_Loss:2.09879 Test_Loss:2.14655 Train_Acc:0.31250 Test_Acc:0.28046 gap:0.03204 Train_Speed:1064.5272759 Test_Speed:4004.49709474
best_epoch:10 best_test_acc:0.299446202532
E:11 S:4600 Train_Loss:2.43194 Test_Loss:2.16936 Train_Acc:0.27344 Test_Acc:0.28066 gap:-0.00722 Train_Speed:1010.70800835 Test_Speed:3921.22728136
best_epoch:10 best_test_acc:0.299446202532
E:11 S:4650 Train_Loss:2.31174 Test_Loss:2.19297 Train_Acc:0.34375 Test_Acc:0.28748 gap:0.05627 Train_Speed:1081.38069725 Test_Speed:4239.54792552
best_epoch:10 best_test_acc:0.299446202532
E:12 S:4700 Train_Loss:2.87508 Test_Loss:2.36022 Train_Acc:0.28125 Test_Acc:0.27275 gap:0.00850 Train_Speed:1180.24734435 Test_Speed:1132.95456119
best_epoch:10 best_test_acc:0.299446202532
E:12 S:4750 Train_Loss:2.18191 Test_Loss:2.26949 Train_Acc:0.31250 Test_Acc:0.28590 gap:0.02660 Train_Speed:1057.97378274 Test_Speed:4217.86472876
best_epoch:10 best_test_acc:0.299446202532
E:12 S:4800 Train_Loss:1.92283 Test_Loss:2.23566 Train_Acc:0.32031 Test_Acc:0.28352 gap:0.03679 Train_Speed:1215.3549871 Test_Speed:4259.02115743
best_epoch:10 best_test_acc:0.299446202532
E:12 S:4850 Train_Loss:2.24625 Test_Loss:2.14552 Train_Acc:0.21094 Test_Acc:0.28867 gap:-0.07773 Train_Speed:1131.89401069 Test_Speed:4383.26376121
best_epoch:10 best_test_acc:0.299446202532
E:12 S:4900 Train_Loss:2.12096 Test_Loss:2.23744 Train_Acc:0.27344 Test_Acc:0.29242 gap:-0.01899 Train_Speed:1141.6955427 Test_Speed:3354.75128255
best_epoch:10 best_test_acc:0.299446202532
E:12 S:4950 Train_Loss:2.29347 Test_Loss:2.22825 Train_Acc:0.26562 Test_Acc:0.28441 gap:-0.01879 Train_Speed:1160.74424839 Test_Speed:3632.85726272
best_epoch:10 best_test_acc:0.299446202532
E:12 S:5000 Train_Loss:2.23441 Test_Loss:2.16348 Train_Acc:0.25000 Test_Acc:0.28669 gap:-0.03669 Train_Speed:1190.02381063 Test_Speed:4523.45611108
best_epoch:10 best_test_acc:0.299446202532
E:12 S:5050 Train_Loss:2.30547 Test_Loss:2.17714 Train_Acc:0.27344 Test_Acc:0.29707 gap:-0.02364 Train_Speed:957.589988727 Test_Speed:4973.00695647
best_epoch:10 best_test_acc:0.299446202532
E:13 S:5100 Train_Loss:2.34201 Test_Loss:2.23939 Train_Acc:0.25000 Test_Acc:0.27008 gap:-0.02008 Train_Speed:1159.12531279 Test_Speed:4573.08397077
best_epoch:10 best_test_acc:0.299446202532
E:13 S:5150 Train_Loss:2.31656 Test_Loss:2.16713 Train_Acc:0.28125 Test_Acc:0.29371 gap:-0.01246 Train_Speed:927.912890029 Test_Speed:3734.15670536
best_epoch:10 best_test_acc:0.299446202532
E:13 S:5200 Train_Loss:2.24781 Test_Loss:2.19491 Train_Acc:0.28125 Test_Acc:0.29312 gap:-0.01187 Train_Speed:1197.11132914 Test_Speed:4471.91190631
best_epoch:10 best_test_acc:0.299446202532
E:13 S:5250 Train_Loss:2.11684 Test_Loss:2.09564 Train_Acc:0.32031 Test_Acc:0.30380 gap:0.01652 Train_Speed:1194.64198185 Test_Speed:4327.68459151
best_epoch:13 best_test_acc:0.303797468354
E:13 S:5300 Train_Loss:2.12072 Test_Loss:2.11669 Train_Acc:0.28906 Test_Acc:0.30113 gap:-0.01206 Train_Speed:1217.7119022 Test_Speed:4333.55325418
best_epoch:13 best_test_acc:0.303797468354
E:13 S:5350 Train_Loss:2.33529 Test_Loss:2.15970 Train_Acc:0.29688 Test_Acc:0.28511 gap:0.01177 Train_Speed:1158.78005728 Test_Speed:4201.26234075
best_epoch:13 best_test_acc:0.303797468354
E:13 S:5400 Train_Loss:2.37300 Test_Loss:2.07269 Train_Acc:0.26562 Test_Acc:0.30953 gap:-0.04391 Train_Speed:1079.58453551 Test_Speed:4171.68564191
best_epoch:13 best_test_acc:0.309533227848
E:13 S:5450 Train_Loss:1.91987 Test_Loss:2.06280 Train_Acc:0.35156 Test_Acc:0.30538 gap:0.04618 Train_Speed:912.336690701 Test_Speed:4424.29838643
best_epoch:13 best_test_acc:0.309533227848
E:14 S:5500 Train_Loss:2.27801 Test_Loss:2.11238 Train_Acc:0.26562 Test_Acc:0.30231 gap:-0.03669 Train_Speed:1053.17748052 Test_Speed:4062.58730231
best_epoch:13 best_test_acc:0.309533227848
E:14 S:5550 Train_Loss:1.97383 Test_Loss:2.08985 Train_Acc:0.30469 Test_Acc:0.29826 gap:0.00643 Train_Speed:1172.31217642 Test_Speed:4422.36683992
best_epoch:13 best_test_acc:0.309533227848
E:14 S:5600 Train_Loss:2.49850 Test_Loss:2.10744 Train_Acc:0.28125 Test_Acc:0.29905 gap:-0.01780 Train_Speed:1137.51549258 Test_Speed:4432.59036155
best_epoch:13 best_test_acc:0.309533227848
E:14 S:5650 Train_Loss:2.00086 Test_Loss:2.15149 Train_Acc:0.35156 Test_Acc:0.29826 gap:0.05330 Train_Speed:1198.23081644 Test_Speed:4246.01918681
best_epoch:13 best_test_acc:0.309533227848
E:14 S:5700 Train_Loss:2.08668 Test_Loss:2.06769 Train_Acc:0.34375 Test_Acc:0.29974 gap:0.04401 Train_Speed:1185.27109274 Test_Speed:4277.78770059
best_epoch:13 best_test_acc:0.309533227848
E:14 S:5750 Train_Loss:2.26744 Test_Loss:2.04210 Train_Acc:0.31250 Test_Acc:0.30953 gap:0.00297 Train_Speed:1186.84323713 Test_Speed:4899.17243393
best_epoch:13 best_test_acc:0.309533227848
E:14 S:5800 Train_Loss:2.42013 Test_Loss:2.05375 Train_Acc:0.25000 Test_Acc:0.30063 gap:-0.05063 Train_Speed:1212.24574991 Test_Speed:4548.21172484
best_epoch:13 best_test_acc:0.309533227848
E:14 S:5850 Train_Loss:2.10737 Test_Loss:2.07919 Train_Acc:0.33594 Test_Acc:0.30696 gap:0.02898 Train_Speed:1065.12073676 Test_Speed:4101.38206264
best_epoch:13 best_test_acc:0.309533227848
E:15 S:5900 Train_Loss:2.12405 Test_Loss:2.08203 Train_Acc:0.33594 Test_Acc:0.29272 gap:0.04322 Train_Speed:1091.60928477 Test_Speed:4205.4089079
best_epoch:13 best_test_acc:0.309533227848
E:15 S:5950 Train_Loss:2.24514 Test_Loss:2.05870 Train_Acc:0.30469 Test_Acc:0.29579 gap:0.00890 Train_Speed:1156.53093441 Test_Speed:4327.68459151
best_epoch:13 best_test_acc:0.309533227848
E:15 S:6000 Train_Loss:2.03245 Test_Loss:2.09055 Train_Acc:0.31250 Test_Acc:0.28797 gap:0.02453 Train_Speed:1155.93330577 Test_Speed:4050.38862902
best_epoch:13 best_test_acc:0.309533227848
E:15 S:6050 Train_Loss:2.24727 Test_Loss:2.10345 Train_Acc:0.32031 Test_Acc:0.30775 gap:0.01256 Train_Speed:1166.74543621 Test_Speed:4533.99976353
best_epoch:13 best_test_acc:0.309533227848
E:15 S:6100 Train_Loss:1.92576 Test_Loss:2.07067 Train_Acc:0.28125 Test_Acc:0.29688 gap:-0.01562 Train_Speed:1136.7471728 Test_Speed:4419.41810998
best_epoch:13 best_test_acc:0.309533227848
E:15 S:6150 Train_Loss:2.08536 Test_Loss:2.05159 Train_Acc:0.33594 Test_Acc:0.31477 gap:0.02116 Train_Speed:1177.81341758 Test_Speed:4039.90392198
best_epoch:15 best_test_acc:0.314774525316
E:15 S:6200 Train_Loss:1.90426 Test_Loss:2.09866 Train_Acc:0.29688 Test_Acc:0.29529 gap:0.00158 Train_Speed:1219.97539466 Test_Speed:4544.93893757
best_epoch:15 best_test_acc:0.314774525316
E:15 S:6250 Train_Loss:2.19895 Test_Loss:2.07545 Train_Acc:0.28906 Test_Acc:0.30133 gap:-0.01226 Train_Speed:1187.86999763 Test_Speed:3007.37691437
best_epoch:15 best_test_acc:0.314774525316
E:16 S:6300 Train_Loss:2.09298 Test_Loss:2.06988 Train_Acc:0.25000 Test_Acc:0.29153 gap:-0.04153 Train_Speed:1104.79086651 Test_Speed:2219.63969521
best_epoch:15 best_test_acc:0.314774525316
E:16 S:6350 Train_Loss:2.27335 Test_Loss:2.03631 Train_Acc:0.25781 Test_Acc:0.30439 gap:-0.04658 Train_Speed:1195.23507935 Test_Speed:4413.7864266
best_epoch:15 best_test_acc:0.314774525316
E:16 S:6400 Train_Loss:2.01116 Test_Loss:2.04440 Train_Acc:0.27344 Test_Acc:0.29529 gap:-0.02186 Train_Speed:1213.38550865 Test_Speed:3909.83309665
best_epoch:15 best_test_acc:0.314774525316
E:16 S:6450 Train_Loss:1.89976 Test_Loss:2.05661 Train_Acc:0.34375 Test_Acc:0.30597 gap:0.03778 Train_Speed:884.8873879 Test_Speed:3824.87487443
best_epoch:15 best_test_acc:0.314774525316
E:16 S:6500 Train_Loss:1.93772 Test_Loss:2.03792 Train_Acc:0.32031 Test_Acc:0.29935 gap:0.02097 Train_Speed:1203.35791135 Test_Speed:4516.11228224
best_epoch:15 best_test_acc:0.314774525316
E:16 S:6550 Train_Loss:2.47803 Test_Loss:2.05896 Train_Acc:0.26562 Test_Acc:0.30172 gap:-0.03610 Train_Speed:1062.20416239 Test_Speed:3959.89667864
best_epoch:15 best_test_acc:0.314774525316
E:16 S:6600 Train_Loss:2.27755 Test_Loss:2.01047 Train_Acc:0.24219 Test_Acc:0.30845 gap:-0.06626 Train_Speed:1142.23479101 Test_Speed:4090.22689837
best_epoch:15 best_test_acc:0.314774525316
E:17 S:6650 Train_Loss:2.56299 Test_Loss:2.43982 Train_Acc:0.27344 Test_Acc:0.27888 gap:-0.00544 Train_Speed:973.91018643 Test_Speed:3597.82411323
best_epoch:15 best_test_acc:0.314774525316
E:17 S:6700 Train_Loss:1.98020 Test_Loss:2.07314 Train_Acc:0.25781 Test_Acc:0.29717 gap:-0.03936 Train_Speed:1261.96667795 Test_Speed:4582.7258154
best_epoch:15 best_test_acc:0.314774525316
E:17 S:6750 Train_Loss:2.05947 Test_Loss:2.04610 Train_Acc:0.28906 Test_Acc:0.30182 gap:-0.01276 Train_Speed:1256.0323044 Test_Speed:4453.88179857
best_epoch:15 best_test_acc:0.314774525316
E:17 S:6800 Train_Loss:1.94956 Test_Loss:2.03221 Train_Acc:0.32812 Test_Acc:0.30103 gap:0.02710 Train_Speed:1281.53810091 Test_Speed:4259.83220001
best_epoch:15 best_test_acc:0.314774525316
E:17 S:6850 Train_Loss:1.93869 Test_Loss:2.05692 Train_Acc:0.33594 Test_Acc:0.30508 gap:0.03085 Train_Speed:1217.63181028 Test_Speed:4405.31153944
best_epoch:15 best_test_acc:0.314774525316
E:17 S:6900 Train_Loss:2.07925 Test_Loss:2.03596 Train_Acc:0.28906 Test_Acc:0.29341 gap:-0.00435 Train_Speed:1283.28416161 Test_Speed:1913.09847521
best_epoch:15 best_test_acc:0.314774525316
E:17 S:6950 Train_Loss:2.03528 Test_Loss:2.00767 Train_Acc:0.28125 Test_Acc:0.30696 gap:-0.02571 Train_Speed:1282.26964231 Test_Speed:4485.06217106
best_epoch:15 best_test_acc:0.314774525316
E:17 S:7000 Train_Loss:1.76588 Test_Loss:2.01119 Train_Acc:0.38281 Test_Acc:0.30746 gap:0.07536 Train_Speed:948.857053983 Test_Speed:3732.75471226
best_epoch:15 best_test_acc:0.314774525316
E:18 S:7050 Train_Loss:2.37053 Test_Loss:2.15638 Train_Acc:0.34375 Test_Acc:0.29035 gap:0.05340 Train_Speed:1263.3624157 Test_Speed:4704.52436951
best_epoch:15 best_test_acc:0.314774525316
E:18 S:7100 Train_Loss:1.94380 Test_Loss:2.05977 Train_Acc:0.32031 Test_Acc:0.29836 gap:0.02195 Train_Speed:1201.46204526 Test_Speed:3903.38019485
best_epoch:15 best_test_acc:0.314774525316
E:18 S:7150 Train_Loss:2.10513 Test_Loss:2.04257 Train_Acc:0.26562 Test_Acc:0.30103 gap:-0.03540 Train_Speed:1274.54664847 Test_Speed:5034.18736931
best_epoch:15 best_test_acc:0.314774525316
E:18 S:7200 Train_Loss:2.02944 Test_Loss:2.00461 Train_Acc:0.33594 Test_Acc:0.30756 gap:0.02838 Train_Speed:1299.57084969 Test_Speed:3688.64292634
best_epoch:15 best_test_acc:0.314774525316
E:18 S:7250 Train_Loss:1.96929 Test_Loss:2.06279 Train_Acc:0.31250 Test_Acc:0.30093 gap:0.01157 Train_Speed:1328.24071431 Test_Speed:4367.71597325
best_epoch:15 best_test_acc:0.314774525316
E:18 S:7300 Train_Loss:2.02918 Test_Loss:2.00739 Train_Acc:0.30469 Test_Acc:0.30113 gap:0.00356 Train_Speed:1311.75762077 Test_Speed:3894.12195812
best_epoch:15 best_test_acc:0.314774525316
E:18 S:7350 Train_Loss:2.13662 Test_Loss:1.99577 Train_Acc:0.33594 Test_Acc:0.31260 gap:0.02334 Train_Speed:1302.78433856 Test_Speed:4919.82434661
best_epoch:15 best_test_acc:0.314774525316
E:18 S:7400 Train_Loss:1.94479 Test_Loss:2.01498 Train_Acc:0.32812 Test_Acc:0.31181 gap:0.01632 Train_Speed:761.886458718 Test_Speed:4576.82658437
best_epoch:15 best_test_acc:0.314774525316
E:19 S:7450 Train_Loss:2.11968 Test_Loss:2.06430 Train_Acc:0.35156 Test_Acc:0.30756 gap:0.04401 Train_Speed:1241.10193816 Test_Speed:4498.4784616
best_epoch:15 best_test_acc:0.314774525316
early stop

```



