from utils.header import *
from packages.engine.base_engine import BaseNetwork

class TeacherNetwork():
    '''
    Comment
    '''
    def __init__(self,
                samples_region_train,
                samples_region_test,
                samples_base_train,
                sample_base_test,
                student,
                distillation_loss_fn,
                model_path,
                region,
                num_classes):
        self.loadConfigFile()
        self.samples_region_train = samples_region_train
        self.samples_region_test = samples_region_test
        self.samples_base_train = samples_base_train
        self.sample_base_test = sample_base_test
        self.student = student
        self.distillation_loss_fn = distillation_loss_fn
        self.total_clients = len(samples_region_train)
        self.selected_clients = int(self.total_clients*self.fraction)
        self.regional_model = BaseNetwork()
        self.regional_model.predictor.compile(optimizer='adam',
                                                    loss='categorical_crossentropy', 
                                                    metrics=['accuracy'])
        if self.verbose == True:
            self.regional_model.predictor.summary()
        # self.cluster_test_images, self.cluster_test_labels = self.samples_region_test[0]
        self.model_path = model_path
        self.region = region
        self.teacher_params = self.regional_model.predictor.get_weights()
        self.num_classes = num_classes

    def loadConfigFile(self):
        # open configure file
        config_path = str(from_root("system_configuration","configure.yaml"))
        with open(config_path) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            for doc in docs:
                for k, v in doc.items():
                    if k == "log_level":
                        self.log_level = v
                    if k == "engine":
                        for i, j in v.items():
                            if i == "logger_name":
                                self.logger_name = j
                            if i == "teacher_nw":
                                for m, n in j.items():
                                    if m == "training_round":
                                        self.training_round = n
                                    if m == "batch_size":
                                        self.batch_size = n
                                    if m == "epochs":
                                        self.epochs = n
                                    if m == "fraction":
                                        self.fraction = n
                                    if m == "verbose":
                                        self.verbose = n
                                    if m == "shuffle_per_round":
                                        self.shuffle_per_round = n
        

    def regionalAggregation(self):
        # starting to training
        selected_clients_list = np.random.choice(self.total_clients, size=self.selected_clients, replace=False)

        list_of_local_parameter=[]
        list_of_local_dataset_size=[]
        list_of_local_accuracy=[]
        list_of_local_loss=[]
        for round in range(self.training_round):
            if self.verbose == True: 
                print("\n▶ Round", round+1, "◀")

            # check whether to apply shuffle mode per round
            if self.shuffle_per_round == True:
                selected_clients_list = np.random.choice(self.total_clients, size=self.selected_clients, replace=False)

            for client_idx in selected_clients_list:
                # get data for each client
                train_images, train_labels = self.samples_region_train[client_idx]
                # update teacher params
                self.regional_model.predictor.set_weights(self.teacher_params)

                train_result = self.regional_model.predictor.fit(train_images,
                                                                    train_labels,
                                                                    self.batch_size,
                                                                    self.epochs)
                list_of_local_parameter.append(self.regional_model.predictor.get_weights())
                list_of_local_dataset_size.append(len(train_images))
                list_of_local_accuracy.append(train_result.history["accuracy"][-1])
                list_of_local_loss.append(train_result.history["loss"][-1])
            
            current_mean_accuracy = np.mean(np.array(list_of_local_accuracy, dtype="float32"))
            current_mean_loss = np.mean(np.array(list_of_local_loss, dtype="float32"))

            if self.verbose == True:
                print(f"  evaluation mean : accuracy - {current_mean_accuracy}, loss - {current_mean_loss}")

            list_of_local_parameter.clear()
            list_of_local_dataset_size.clear()
            list_of_local_accuracy.clear()
            list_of_local_loss.clear()

            # evaluate_info = self.regional_model.predictor.evaluate(self.cluster_test_images, self.cluster_test_labels, batch_size=20, verbose=0)
            # if verbose == True: 
            #   print(f"  Weight Validation :  accuracy - {evaluate_info[1]}, loss - {evaluate_info[0]}")

        self.save_path = os.path.join(self.model_path,f"teacher_region_{self.region}.h5")
        self.regional_model.predictor.save(self.save_path)
        self.teacher_params = np.mean(list_of_local_parameter, axis=0)
        return self.teacher_params, self.save_path

class StudentNetwork():
    def __init__(self):
        self.model = BaseNetwork().predictor
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])

