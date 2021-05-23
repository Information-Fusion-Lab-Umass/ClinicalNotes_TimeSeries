import os
class Config():
    def __init__(self):
        self.basepath = "D:/_Projects/ClinicalNotes_TimeSeries/ClinicalNotesICU-master_pytorch/"
        self.data = 'D:/_Projects/DeepFusion/Data/MIMIC/MIMIC_Last/MIMIC/in-hospital-mortality/'
        self.timestep = 1.0
        self.normalizer_state = "D:/_Projects/DeepFusion/Data/MIMIC/mimic3-benchmarks/mimic3models/in_hospital_mortality/ihm_ts10"
        self.imputation = 'previous'
        self.small_part = False
        self.textdata = self.basepath + 'text/'
        self.embeddingspath = 'D:/_Projects/DeepFusion/Data/MIMIC/mimic3-benchmarks/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
        self.buffer_size = 100
        self.model_path = os.path.join(self.basepath, 'wv.pkl')
        self.learning_rate = 2e-5
        self.max_len = 128
        self.break_text_at = 300
        self.padding_type = 'Zero'
        self.los_path = "D:/_Projects/DeepFusion/Data/MIMIC/mimic3-benchmarks/mimic3models/length_of_stay"
        self.decompensation_path = "D:/_Projects/DeepFusion/Data/MIMIC/mimic3-benchmarks/mimic3models/decompensation"
        self.ihm_path = "D:/_Projects/DeepFusion/Data/MIMIC/MIMIC_Last/MIMIC/in-hospital-mortality/"
        self.textdata_fixed = "D:/_Projects/DeepFusion/Data/MIMIC/MIMIC_Last/MIMIC/root/text_fixed/"
        self.multitask_path = "D:/_Projects/DeepFusion/Data/MIMIC/mimic3-benchmarks/mimic3models/multitask"
        self.starttime_path = "D:/_Projects/DeepFusion/Data/MIMIC/MIMIC_Last/MIMIC/starttime.pkl"
        self.rnn_hidden_units = 256
        self.maximum_number_events = 150
        self.conv1d_channel_size = 256
        self.test_textdata_fixed = "D:/_Projects/DeepFusion/Data/MIMIC/MIMIC_Last/MIMIC/root/test_text_fixed/"
        self.test_starttime_path = "D:/_Projects/DeepFusion/Data/MIMIC/MIMIC_Last/MIMIC/test_starttime.pkl"
        self.dropout = 0.9 #keep_prob
        # Not used in final mode, just kept for reference.
        self.trainpicklepath = self.basepath + 'train.pkl'
        self.evalpicklepath = self.basepath + 'val.pkl'
        self.patient2hadmid_picklepath = os.path.join(self.basepath,'patient2hadmid.pkl')
        self.trainpicklepath_new = os.path.join(self.basepath , 'train_text_ts.pkl')
        self.evalpicklepath_new = os.path.join(self.basepath , 'val_text_ts.pkl')
        self.num_blocks = 3
        self.mortality_class_ce_weigth = 10