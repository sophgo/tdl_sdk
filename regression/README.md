# TDLSDK Regression
### _前置作業 交叉編譯 (執行檔 & 庫文件 生成)_
1. 開發版環境編譯 (以 cv1835_wevb_0002a 為範例)
    ```sh
    # 進入SDK Package(此路徑由使用者決定，這裡以sdk_package 當其路徑)
    cd sdk_package
    source build/envsetup_soc.sh
    # 選取板子相對應的設定
    defconfig cv1835_wevb_0002a
    # TPU_REL=1 開啟 TPU 相關編譯，包含 tpu、ive、ivs、ai
    export TPU_REL=1
    # 第一次需要 build_all
    clean_all
    build_all

    # build_all 之後如需要對cvitdl 做單獨編譯
    build_ai_sdk
    ```
2. 確認TDLSDK 生成路徑如下：
    sdk_package/install/soc_cv1835_wevb_0002a/tpu_64bit/cvitek_ai_sdk/

### _Regression 環境/運行_
1. 準備FAT32 格式的SD 卡，建立以下路徑，並放置檔案
    | Destination file | Source file |
    | ------- | ------ |
    | bin | sdk_package/install/soc_cv1835_wevb_0002a/tpu_$SDK_VER/cvitek_ai_sdk/regression/* |
    | lib | sdk_package/install/soc_cv1835_wevb_0002a/tpu_64bit/cvitek_ai_sdk/lib <br> sdk_package/install/soc_cv1835_wevb_0002a/tpu_64bit/cvitek_ive_sdk/lib <br> sdk_package/install/soc_cv1835_wevb_0002a/tpu_64bit/cvitek_tpu_sdk/lib <br> sdk_package/middleware/$MW_VER/lib/3rd <br> sdk_package/middleware/$MW_VER/lib   |
    | dataset | git clone ssh://shiqi.lin@gerrit-ai.sophgo.vip:29418/cvitek/aisdk_daily_regression |
    | assets | sdk_package/install/soc_cv1835_wevb_0002a/tpu_64bit/cvitek_ai_sdk/regression/assets |
    | ai_models_output | Retrieve data from the daily build FTP (10.80.0.5) /sw_rls/ai_models/xxxx-xx-xx/ai_models.tar.gz |
2. 將準備好的SD 卡，掛載到板子上，準備運行
    ```sh
    mount /dev/xxx /mnt/sd
    export LD_LIBRARY_PATH=/mnt/sd/lib
    cd /mnt/sd/bin
    # CHIP_SEGMENT [cv180x/cv181x/cv182x/cv183x]
    # 1. Run executable file
    ./test_main /mnt/sd/ai_models_output/$CHIP_SEGMENT /mnt/sd/dataset /mnt/sd/assets/
    # 2. Run script
    ./daily_regression.sh
    # 3. Run Single
    ./test_main /mnt/sd/ai_models_output/$CHIP_SEGMENT /mnt/sd/dataset /mnt/sd/assets/ --gtest_filter=FaceRecognitionTestSuite.*
    ```
3. 測試程式對應相關文檔
    | Regression code | Json file | Cvimodel | Image folder |
    | ------ | ------| ------ | ------ |
    |reg_daily_alphapose.cpp|reg_daily_alphapose.json|alphapose.cvimodel|reg_daily_alphapose|
    |reg_daily_core.cpp | ----- | ----- | -----|
    |reg_daily_es_classification.cpp|reg_daily_es_classification.json|es_classification.cvimodel|reg_daily_es_classification|
    |reg_daily_eye_classification.cpp|reg_daily_eye_classification.jsoneye_v1_bf16.cvimodel|eye_v1_bf16.cvimodel|reg_daily_eye_classification|
    |reg_daily_fall.cpp|reg_daily_fall.json|ive|reg_daily_fall|
    |reg_daily_fdmask.cpp|reg_daily_fdmask.json|retinaface_yolox_fdmask.cvimodel|reg_daily_fdmask|
    |reg_daily_feature_matching.cpp| ----- | ----- | -----|
    |reg_daily_fl.cpp|reg_daily_fl.json|face_landmark_bf16.cvimodel|reg_daily_fl|
    |reg_daily_fq.cpp|daily_reg_FQ.json|fqnet-v5_shufflenetv2-softmax.cvimodel|reg_daily_fq|
    |reg_daily_fr.cpp|daily_reg_FR.json|cviface-v5-s.cvimodel|reg_daily_fr|
    |reg_daily_incarod.cpp|reg_daily_incarod.json|incar_od_v0_bf16.cvimodel|reg_daily_incarod|
    |reg_daily_liveness.cpp|reg_daily_liveness.json|liveness-rgb-ir.cvimodel|reg_daily_liveness|
    |reg_daily_lpd.cpp|daily_reg_LPD.json|wpodnet_v0_bf16.cvimodel|reg_daily_lpd|
    |reg_daily_lpr.cpp|daily_reg_LPR.json|lprnet_v0_tw_bf16.cvimodel|reg_daily_lpr|
    |reg_daily_mask_classification.cpp|daily_reg_MaskClassification.json|mask_classifier.cvimodel|reg_daily_mask_classification|
    |reg_daily_md.cpp|daily_reg_md.json|ive|reg_daily_md|
    |reg_daily_mobiledet.cpp|daily_reg_mobiledet.json|mobiledetv2-d0-ls.cvimodel mobiledetv2-pedestrian-d0-ls.cvimodel mobiledetv2-pedestrian-d0-ls-640.cvimodel mobiledetv2-lite-person-pets-ls.cvimodel mobiledetv2-lite-person-pets.cvimodel mobiledetv2-person-vehicle-ls.cvimodel mobiledetv2-pedestrian-d0.cvimodel obiledetv2-d1-ls.cvimodel mobiledetv2-person-vehicle-ls-768.cvimodel mobiledetv2-vehicle-d0-ls.cvimodel mobiledetv2-pedestrian-d0-ls-384.cvimodel mobiledetv2-d2.cvimodel mobiledetv2-vehicle-d0.cvimodel mobiledetv2-d2-ls.cvimodel mobiledetv2-pedestrian-d1-ls.cvimodel mobiledetv2-d0.cvimodel mobiledetv2-pedestrian-d1.cvimodel mobiledetv2-d1.cvimodel mobiledetv2-pedestrian-d1-ls-1024.cvimodel mobiledetv2-person-vehicle.cvimodel mobiledetv2-pedestrian-d0-ls-768.cvimodel|reg_daily_mobildet|
    |reg_daily_mot.cpp|daily_reg_MOT.json|ive|reg_daily_mot|
    |reg_daily_reid.cpp|daily_reg_ReID.json|person-reid-v1.cvimodel|reg_daily_reid|
    |reg_daily_retinaface.cpp|reg_daily_retinaface.json|retinaface_mnet0.25_608.cvimodel retinaface_mnet0.25_608_342.cvimodel retinaface_mnet0.25_342_608.cvimodel|reg_daily_retinaface|
    |reg_daily_retinafaceIR.cpp|reg_daily_retinafaceIR.json|retinafaceIR_mnet0.25_608_608.cvimodel retinafaceIR_mnet0.25_608_342.cvimodel retinafaceIR_mnet0.25_342_608.cvimodel|reg_daily_retinafaceIR|
    |reg_daily_retinaface_hardhat.cpp|reg_daily_retinaface_hardhat.json|hardhat_720_1280.cvimodel|reg_daily_retinaface_hardhat
    |reg_daily_soundcmd.cpp|reg_daily_soundcmd.json|soundcmd_bf16.cvimodel|reg_daily_soundcmd|
    |reg_daily_td.cpp|daily_reg_TD.json|ive|reg_daily_td|
    |reg_daily_thermal_fd.cpp|daily_reg_ThermalFD.json|thermalfd-v1.cvimodel|reg_daily_thermal_fd|
    |reg_daily_thermal_person_detection.cpp|reg_daily_thermal_person_detection.json|thermal_person_detection.cvimodel|reg_daily_thermal_person_detection|
    |reg_daily_yawn_classification.cpp|reg_daily_yawn_classification.json|yawn_v1_bf16.cvimodel|reg_daily_yawn_classification|
    |reg_daily_face_cap.cpp|daily_reg_face_cap.json|scrfd_500m_bnkps_432_768.cvimodel cviface-v5-s.cvimodel mobiledetv2-pedestrian-d0-ls-448.cvimodel pipnet_blurness_v5_64_retinaface_50ep.cvimodel|reg_daily_face_cap|
