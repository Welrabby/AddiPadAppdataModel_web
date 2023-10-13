import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

st.title('問題行動予測システム')

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください。",type='csv')

if uploaded_file is not None:


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++データの読み込み++++++++++++++++++++++++++++++++++++++++++++++++++++++
    df_data = pd.read_csv(
        uploaded_file,
        encoding='shift-jis',
        engine='python',
        na_values='-',
        header=0) 
    try:  
        # indexカラムの追加
        df_data = df_data.reset_index()

        # 型変換
        df_data['StudentGender'] = df_data['StudentGender'].astype(str)
        df_data['SchoolGrade'] = df_data['SchoolGrade'].astype(str)
        df_data['gakku'] = df_data['gakku'].astype(str)

        # One Hot Encoding
        df_data_one_hot = pd.get_dummies(df_data, columns=['StudentGender', 'SchoolGrade', 'gakku'])

        # フラッグ処理**********************************************************************************
        df_data_one_hot = df_data_one_hot[df_data_one_hot['flag1（前後重複70%以上）'] != 1] #機械学習時除外
        df_data_one_hot = df_data_one_hot[df_data_one_hot['flag2（欠測有無）'] != 1] #機械学習時除外
        df_data_one_hot = df_data_one_hot[df_data_one_hot['flag3（保護者不同意）'] != 1] #機械学習時除外
        df_data_one_hot = df_data_one_hot[df_data_one_hot['flag4（デスクトップPC実施）'] != 1] #機械学習時除外

        # 以下も使わない場合はコメントを外してください．
        # df_data_one_hot = df_data_one_hot[df_data_one_hot['flag5（特支学級）'] != 1] 
        # df_data_one_hot = df_data_one_hot[df_data_one_hot['flag6（IAT誤答率20%以上）'] != 1] 
        # ******************************************************************************************

        #IATの'.'をNaNに変更しておく。
        df_data_preprocessing = df_data_one_hot.replace({'IAT': {'.': np.nan}})

        # IAT欠損値の削除。 IATを削除するとpoやneやerrorも自動的になくなる．
        df_data_preprocessing = df_data_preprocessing.dropna(subset = ['IAT'])

        # 内面指標〜環境指標
        columns = df_data_preprocessing.loc[:, 'sc1':'social_cohesion'].columns

        # 欠損値のある行の削除
        df_data_preprocessing = df_data_preprocessing.dropna(subset=columns)

        # リセットインデックス
        df_data_preprocessing = df_data_preprocessing.reset_index()

        # 予測に使う部分をピックアップする
        df_data_predict = pd.concat([df_data_preprocessing.loc[:,'StudentGender_1':'StudentGender_2'], #性別ラベル
                            df_data_preprocessing.loc[:, 'SchoolGrade_4':'SchoolGrade_9'], #学年ラベル
                            # df_data_preprocessing.loc[:,'gakku_1':'gakku_9'], #学区
                            df_data_preprocessing.loc[:,'IAT_po':'IAT2_error'], #IAT
                            df_data_preprocessing.loc[:,'sc1':'ps10'], #内面指標と環境指標の詳細カテゴリ
                            df_data_preprocessing.loc[:,'self_control':'social_cohesion'], #内面指標と環境指標
                            df_data_preprocessing.loc[:,'hostile_mentality':'Standardized_T_CorrectAnswer'] #今回追加するカラム
                            ], axis=1)
        
        # IATの型を変換する
        li = ['IAT_po', 'IAT_ne', 'IAT1_error', 'IAT2_error']
        for col in li:
            df_data_predict[col] = df_data_predict[col].astype(np.float64)
            
        Na = ['hostile_mentality', 'reflections_violence', 'reflections_society']
        for col in Na:
            # '.' を欠損値に置き換える
            df_data_predict[col] = df_data_predict[col].replace('.', np.nan)
            # 欠損値を0.0に置き換える
            df_data_predict[col] = df_data_predict[col].fillna(0.0)
        

        # +++++++++++++++++++++++++++++++++++++++モデルを読み込む+++++++++++++++++++++++++++++++++++++++
    
        folderpath = f'./streamlit/UsingModels' #モデルが格納されたフォルダ
        pkl_files = [file for file in os.listdir(folderpath) if file.lower().endswith('.pkl')]
        
        models_list = [] # モデルを格納するリストを用意

        # models_listのリストの中に保存してあるモデルを格納していく．
        for pklfile in pkl_files:
            filename =f"{folderpath}/{pklfile}"
            model = pickle.load(open(filename, 'rb'))
            models_list.append(model)


        # +++++++++++++++++++++++++++++++++++++++予測+++++++++++++++++++++++++++++++++++++++
        y_preds = [] #予測ラベル（0か1）のリスト
        y_preds_prob = [] #予測確率のリスト

        # 実際に予測させる
        for model in models_list:
            y_preds.append(model.predict(df_data_predict))
            y_preds_prob.append(model.predict_proba(df_data_predict))   

        
        # 最終ラベルを決定する作業
        for index, value in enumerate(y_preds):
            if index == 0:
                df_label_use = pd.DataFrame(value, columns=[f'{index+1}回目']) #初めだけ，データフレーム変数に格納．その後はそこを基準に追加していく
            if index != 0:
                df_ad = pd.DataFrame(value, columns=[f'{index+1}回目'])
                df_label_use = pd.concat([df_label_use, df_ad], axis=1) #結合

        

        df_label_use['ave'] = df_label_use.sum(axis=1) / len(y_preds)  #行ごとで合計値を出して回数でわる．→平均値が出る．


        for index, row in df_label_use.iterrows():
            if row['ave'] >= 0.5:#閾値
                df_label_use.loc[index, 'decided_label'] = 1 #平均が0.5以上ならばラベルを1（陽性）とする
            else:
                df_label_use.loc[index, 'decided_label'] = 0
        decided_label = df_label_use['decided_label'].to_list() #最終ラベルをリストでget
        decided_label = np.array(decided_label) #numpy.array型に変換した。

        # 確率の平均を出す作業
        for index, value in enumerate(y_preds_prob):
            

            if index == 0:
                df_prob_use = pd.DataFrame(value, columns=['陰性（0）', '陽性（1）'])                    
            if index != 0:
                df_ad2 = pd.DataFrame(value, columns=['陰性（0）', '陽性（1）'])
                df_prob_use = pd.concat([df_prob_use, df_ad2], axis=1)
        
        

        df_prob_use['陰性確率（平均）'] = df_prob_use['陰性（0）'].sum(axis=1) / len(y_preds_prob)
        df_prob_use['陽性確率(平均)'] = df_prob_use['陽性（1）'].sum(axis=1) / len(y_preds_prob)
        ave_prob = df_prob_use[['陰性確率（平均）', '陽性確率(平均)']].to_numpy().tolist() #平均確率をリストでget
        ave_prob = np.array(ave_prob) #numpy.array型に変換した。


        #上で作ったラベルのデータフレームと確率のデータフレームを結合する．（最後リターンする）
        df_label_prob = pd.concat([df_label_use, df_prob_use], axis=1)


        #一旦df_dataとdf_label_probを合体してしまう．
        df_predicted = pd.concat([df_data_preprocessing.loc[:, 'index'], df_data_predict, df_label_prob], axis=1)



        df_predicted.loc[df_predicted['陽性確率(平均)'] < 0.15, '7-step_evaluation'] = '①とても安心'
        df_predicted.loc[(df_predicted['陽性確率(平均)'] >= 0.15) & (df_predicted['陽性確率(平均)'] < 0.30), '7-step_evaluation'] = '②安心'
        df_predicted.loc[(df_predicted['陽性確率(平均)'] >= 0.30) & (df_predicted['陽性確率(平均)'] < 0.40), '7-step_evaluation'] = '③安心寄り'
        df_predicted.loc[(df_predicted['陽性確率(平均)'] >= 0.40) & (df_predicted['陽性確率(平均)'] < 0.60), '7-step_evaluation'] = '④曖昧'
        df_predicted.loc[(df_predicted['陽性確率(平均)'] >= 0.60) & (df_predicted['陽性確率(平均)'] < 0.70), '7-step_evaluation'] = '⑤危険寄り'
        df_predicted.loc[(df_predicted['陽性確率(平均)'] >= 0.70) & (df_predicted['陽性確率(平均)'] < 0.85), '7-step_evaluation'] = '⑥危険'
        df_predicted.loc[df_predicted['陽性確率(平均)'] >= 0.85, '7-step_evaluation'] = '⑦とても危険'



        # 基準グループとレッドライングループの読み込み
        df_base_group = pd.read_csv('./Datasets/Input_data/df_base_group.csv')
        df_for_red = pd.read_csv('./Datasets/Input_data/df_for_red.csv')

        df_base_group_stat = df_base_group.loc[:, 'self_control':'social_cohesion'].describe()
        df_for_red_stat = df_for_red.loc[:, 'self_control':'social_cohesion'].describe()

        # 平均 + 2SD
        mean_base = df_base_group_stat.loc['mean', 'self_control':'social_cohesion'].values
        std_base = df_base_group_stat.loc['std', 'self_control':'social_cohesion'].values
        # 平均 + 2SD
        mean_red = df_for_red_stat.loc['mean', 'self_control':'social_cohesion'].values
        std_red = df_for_red_stat.loc['std', 'self_control':'social_cohesion'].values

        mean_2sd_base = []
        mean_2sd_red = []

        for i in range(0, len(df_base_group_stat.columns)):
            mean_2std = mean_base[i] + (2*std_base[i])
            mean_2sd_base.append(mean_2std)

        for i in range(0, len(df_for_red_stat.columns)):
            mean_2std = mean_red[i] + (2*std_red[i])
            mean_2sd_red.append(mean_2std)

        # 評価する
        columns = df_predicted.loc[:, 'self_control':'social_cohesion'].columns
        for idx, col in enumerate(columns):
            df_predicted.loc[df_predicted[col]>mean_2sd_base[idx], f'{col}_evaluation'] = '△'
            df_predicted.loc[df_predicted[col]>mean_2sd_red[idx], f'{col}_evaluation'] = '×'#レッドライン
            

        # ①とても安心，②安心，③安心寄りに関しては詳細フィードバックはしないので，nanで埋めておく．
        for idx, col in enumerate(columns):
            df_predicted.loc[df_predicted['7-step_evaluation']=='①とても安心', f'{col}_evaluation'] = np.nan
            df_predicted.loc[df_predicted['7-step_evaluation']=='②安心', f'{col}_evaluation'] = np.nan
            df_predicted.loc[df_predicted['7-step_evaluation']=='③安心寄り', f'{col}_evaluation'] = np.nan

        # データフレームの完成
        df_predicted_ketugouyou = pd.concat([df_predicted.loc[:, 'index'], df_predicted.loc[:, '陰性確率（平均）':'social_cohesion_evaluation']], axis=1)

        # 予測と評価をもとのdf_dataにくっつける作業を行う．（キーをindexカラムで行う）
        df_comp = pd.merge(df_data, df_predicted_ketugouyou, on="index", how="outer")

        # インデックスカラムを削除する
        df_comp.drop('index', axis=1, inplace=True)


    
        # +++++++++++++++++++++++++++++++++++++++結果の表示+++++++++++++++++++++++++++++++++++++++   
        option = st.selectbox(
        '表示選択',
        ['すべて表示','①とても安心のみ表示','②安心のみ表示','③安心寄りのみ表示','④曖昧のみ表示','⑤危険寄りのみ表示','⑥危険のみ表示']
        )

        st.markdown('### 予測結果')

        
        df_result = df_comp.loc[:, '陰性確率（平均）':'social_cohesion_evaluation']  
        t = df_comp['20_p_num'] 
        df_result = pd.concat([t,df_result],axis = 1)
        
        def selectedoption(optionname):
            #予測結果がとても安心のインデックスを取得
            df_selectedresult = df_result[df_result['7-step_evaluation'] == optionname]
            if(df_selectedresult.empty):
                st.write(f'{optionname} のデータはありません') 
            else:
                st.write(df_selectedresult)     

        if option == 'すべて表示':
            st.write(df_result) 
        elif option == '①とても安心のみ表示':
            selectedoption('①とても安心')
        elif option == '②安心のみ表示':
            selectedoption('②安心')
        elif option == '③安心寄りのみ表示':
            selectedoption('③安心寄り')
        elif option == '④曖昧のみ表示':
            selectedoption('④曖昧')
        elif option == '⑤危険寄りのみ表示':
            selectedoption('⑤危険寄り')
        elif option == '⑥危険のみ表示':
            selectedoption('⑥危険')
    except Exception as e:
        st.error(f"予測時にエラーが発生しました。ファイルが適切か確認してください。Error：{e}")
        st.stop()