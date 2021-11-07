from utils.header import *

class CReliability():
    def __init__(self, base_samples_region_train, num_classes):
        self.loadConfigFile()
        self.base_samples_region_train = base_samples_region_train
        # self.model_path = os.path.join(from_root(), "model")
        self.model_path = "/content/drive/Shareddrives/Duong-Son/saves_regional_model/teacher_20210925080930"
        self.num_classes = num_classes
        # Data Preprocessing
        self.processed_data_x = self.base_samples_region_train[0]
        self.processed_data_y = self.base_samples_region_train[1]

        self.teacher_class_score = self.scoringAUC()

    def loadConfigFile(self):
        # open configure file
        config_path = str(from_root("system_configuration","configure.yaml"))
        with open(config_path) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            for doc in docs:
                for k, v in doc.items():
                    if k == "log_level":
                        self.log_level = v
                    if k == "c_reliability":
                        for i, j in v.items():
                            if i == "logger_name":
                                self.logger_name = j
                            if i == "weighting_temperature":
                                self.weighting_temperature = j
                            if i == "num_regions":
                                self.num_regions = j

    def scoringAUC(self):
        teacher_class_score = []
        for region in range(self.num_regions):
            model_path_name = os.path.join(self.model_path, f'teacher_region_{region}.h5')
            teacher = tf.keras.models.load_model(model_path_name,compile=False)
            predictions = teacher.predict(x=self.processed_data_x, batch_size=32, verbose=False)
            rounded_predictions = np.argmax(predictions, axis=-1)
            rounded_predictions_oh = tf.keras.utils.to_categorical(np.transpose([rounded_predictions]), num_classes=self.num_classes)
            # print(self.processed_data_y.shape)
            # print(rounded_predictions_oh.shape)
            teacher_class_score.append(roc_auc_score(self.processed_data_y, rounded_predictions_oh, average=None, multi_class="ovr"))
        return teacher_class_score
    
    def weightedClass(self):
        teacher_class_score_softmax = []
        teacher_class_score = []
        for i in range(0, len(self.teacher_class_score)):
            teacher_class_score.append(self.teacher_class_score[i])
        teacher_class_score_transpose = np.transpose(teacher_class_score)
        for class_index in range(self.num_classes):
            teacher_class_score_softmax.append(softmax(teacher_class_score_transpose[class_index]*self.weighting_temperature))
        return np.transpose(teacher_class_score_softmax)