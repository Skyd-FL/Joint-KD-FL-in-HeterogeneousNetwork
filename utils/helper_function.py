from utils.header import *

def processDataServer(data):
    normalized_data_x_train_server = data[0].reshape(-1,28,28,1)
    x_base_train = tf.data.Dataset.from_tensor_slices(normalized_data_x_train_server)
    y_base_train = tf.data.Dataset.from_tensor_slices(data[1])
    data_base_train = tf.data.Dataset.zip((x_base_train, y_base_train))
    # dataset_base = data_base_train.shuffle(buffer_size=len(list(data_base_train))).batch(7000)
    dataset_base = data_base_train.shuffle(buffer_size=len(list(data_base_train))).batch(128)
    return dataset_base

def dataAlignment(pseudo_dataset, num_classes):
    # Init aligned_data_teacher
    init_arr = np.array([])
    aligned_data_teacher = {}
    aligned_label_teacher = {}

    for i in range(0,num_classes):
        aligned_data_teacher[i] = init_arr

    for x, y_mu, y_true in pseudo_dataset:
        for i in range(0,num_classes):
            if i == y_mu:
                if aligned_data_teacher[i].size == 0:
                    aligned_data_teacher[i]  = x
                    aligned_label_teacher[i] = y_true
                else:
                    aligned_data_teacher[i]  = np.vstack((aligned_data_teacher[i],x))
                    aligned_label_teacher[i] = np.vstack((aligned_label_teacher[i],y_true))
    return aligned_data_teacher, aligned_label_teacher

class GeneralConfigure():
    def __init__(self):
        self.loadConfigFile()
        self.model_path = os.path.join(from_root(), "model")
        self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy()

    def loadConfigFile(self):
        # open configure file
        config_path = str(from_root("system_configuration","configure.yaml"))
        with open(config_path) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            for doc in docs:
                for k, v in doc.items():
                    if k == "general_configure":
                        for i, j in v.items():
                            if i == "distil_epochs":
                                self.distil_epochs = j
                            if i == "regions":
                                self.regions = j
                            if i == "alpha":
                                self.alpha = j
                            