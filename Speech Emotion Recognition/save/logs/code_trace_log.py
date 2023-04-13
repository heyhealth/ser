"""
2022.12.05: # Add Early Stopping Mechanism to /main/trian.py/train_model()
2022.12.06: # Change to Attentive correlation pooling (papers:SPEECH-BASED EMOTION RECOGNITION WITH SELF-SUPERVISED MODELS USING ATTENTIVE CHANNEL-WISE CORRELATIONS AND LABEL SMOOTHING)
2022.12.09: # Add cls token to Wav2Vec2 in VanillaFineTuningModels4.py  and use IndexPool1D in pooling_v3.py
2022.12.09: # Add aam_softmax as loss
2023.2.16: # finish the experiments in ESD datasets
2023.2.18: # finish the IEMOCAP datasets' preparation
2023.2.22: # Methods : ASR + SER (MultiTaskModels.py,run_IEMOCAP_MT_LOSO.py)
2023.2.24: # Tokens: Word - > Character
2023.3.3: # change some dataset load methods
2023.3.3: # run wav2vec2_v3 on iemocap by random split
"""
