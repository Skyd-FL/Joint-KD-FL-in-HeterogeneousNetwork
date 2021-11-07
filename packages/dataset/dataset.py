from utils.header import *

class JointKDDataset:
    def __init__(self, verbose=False):
        self.loadConfigFile()
        self.images, self.labels = self.downloadEmnist()
        self.check = [0] * self.labels.shape[0]
        # num of clients in each region
        self.num_of_clients = [1, 100, 100, 100]

    def loadConfigFile(self):
        # open configure file
        config_path = str(from_root("system_configuration","configure.yaml"))
        with open(config_path) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            for doc in docs:
                for k, v in doc.items():
                    if k == "log_level":
                        self.log_level = v
                    if k == "dataset":
                        for i, j in v.items():
                            if i == "logger_name":
                                self.logger_name = j
                            if i == "join_kd":
                                for m, n in j.items():
                                    if m == "num_classes":
                                        self.num_classes = n
                                    if m == "num_regions":
                                        self.num_regions = n
                                    if m == "verbose":
                                        self.verbose = n

    def downloadEmnist(self):
        images, labels = extract_training_samples('bymerge')
        if self.verbose == True:
            logging.info(images.shape, labels.shape)
        return images, labels

    def getBalanceDataset(self, no_of_samples):
        x_temp = []
        y_temp = []
        for i in range(0,47):
            count = 0
            for j in range(0, self.labels.shape[0]):
                if self.check[j] == 0:
                    if self.labels[j] == i:
                        x_temp.append(self.images[j])
                        y_temp.append(self.labels[j])
                        count = count + 1
                        self.check[j] = 1
                        if count == no_of_samples:
                            image_array = np.array(x_temp)
                            label_array = np.array(y_temp)
                            break
        return image_array, label_array

    def getUnbalanceDataset(self):
        x_temp = []
        y_temp = []
        x_temp1 = []
        y_temp1 = []
        for i in range(0,47):
            count = 0
            for j in range(0, self.labels.shape[0]):
                if self.check[j] == 0:
                    if self.labels[j] == i:
                        if j%2==0:
                            x_temp.append(self.images[j])
                            y_temp.append(self.labels[j])
                        else:
                            x_temp1.append(self.images[j])
                            y_temp1.append(self.labels[j])
                        count = count + 1
                        self.check[j] = 1
            image_array = np.array(x_temp)
            label_array = np.array(y_temp)
            image_array1 = np.array(x_temp1)
            label_array1 = np.array(y_temp1)
        return image_array, label_array, image_array1, label_array1

    def visualizeData(self, image):
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # plot raw pixel data
            plt.imshow(image[i].reshape(28,28))
            # show the figure
        plt.show()

    def analyzeData(self, data, label):
        data_list = data.tolist()
        label_list = label.tolist()
        input_list = list(zip(data_list, label_list))
        dataset = pd.DataFrame(input_list, columns = ['Data', 'Label'])
        if self.verbose == True:
            print(dataset)
        dataset['Label'].value_counts().plot(kind = 'bar', figsize=(10,10)) # bar, barh, pie, box, area, scatter, hexbin, kde, hist

    def preProcessData(self):
        # Split dataset into 4 parts
        # 1. Server: 300 samples/each class => for training in server
        # 2. Dataset for teacher 1
        # 3. Dataset for teacher 2
        # 4. Dataset for teacher 3
        x1, y1 = self.getBalanceDataset(300)
        x2, y2 = self.getBalanceDataset(1000)
        x3, y3, x4, y4 = self.getUnbalanceDataset()
        # Train-Test
        self.cluster1_X_train, self.cluster1_X_test, self.cluster1_y_train, self.cluster1_y_test = train_test_split(x1, y1, test_size=0.05, random_state=42)
        self.cluster2_X_train, self.cluster2_X_test, self.cluster2_y_train, self.cluster2_y_test = train_test_split(x2, y2, test_size=0.05, random_state=42)
        self.cluster3_X_train, self.cluster3_X_test, self.cluster3_y_train, self.cluster3_y_test = train_test_split(x3, y3, test_size=0.05, random_state=42)
        self.cluster4_X_train, self.cluster4_X_test, self.cluster4_y_train, self.cluster4_y_test = train_test_split(x4, y4, test_size=0.05, random_state=42)
        # Normalize data
        cluster_train_list = [[self.cluster1_X_train, self.cluster1_y_train],
                            [self.cluster2_X_train, self.cluster2_y_train],
                            [self.cluster3_X_train, self.cluster3_y_train],
                            [self.cluster4_X_train, self.cluster4_y_train]]
        cluster_test_list = [[self.cluster1_X_test, self.cluster1_y_test],
                            [self.cluster2_X_test, self.cluster2_y_test],
                            [self.cluster3_X_test, self.cluster3_y_test],
                            [self.cluster4_X_test, self.cluster4_y_test]]
        cluster_data = []
        for i in range(0, len(cluster_train_list)):
            cluster_train = list(zip(cluster_train_list[i][0].reshape(-1, 28, 28, 1).astype("float32")/255.0, cluster_train_list[i][1].astype("float32")))
            cluster_test = list(zip(cluster_test_list[i][0].reshape(-1, 28, 28, 1).astype("float32")/255.0, cluster_test_list[i][1].astype("float32")))
            cluster_data.append((cluster_train, cluster_test))
        return cluster_data

    def assignDataForClients(self, cluster_data):
        # get cluster train data
        list_cluster_train = []
        for i in range(0, len(cluster_data)):
            list_cluster_train.append(cluster_data[i][0])

        # get number of sample in each client - depend on number of clients in region and total number of samples in that region
        num_of_samples = []
        for i in range(0, len(self.num_of_clients)):
            num_of_samples.append(len(cluster_data[i][0])//self.num_of_clients[i]) # 
            print("num_of_clients:", i, " - ", len(cluster_data[i][0])//self.num_of_clients[i])

        list_of_samples_region_train = []
        list_of_samples_region_test = []
        for i in range(0,4):
            temp_list_for_image=[]
            temp_list_for_label=[]
            cluster_federated_train_data_for_iid = []
            for idx, el in enumerate(cluster_data[i][0]):
                temp_list_for_image.append(el[0])
                temp_list_for_label.append(el[1])
                if (idx+1)%(num_of_samples[i])==0:
                    cluster_federated_train_data_for_iid.append((np.array(temp_list_for_image, dtype="float32"), np_utils.to_categorical(np.array(temp_list_for_label, dtype="float32"), 47)))
                    temp_list_for_image=[]
                    temp_list_for_label=[]
            list_of_samples_region_train.append(cluster_federated_train_data_for_iid)
        for i in range(0,4):
            temp_list_for_image=[]
            temp_list_for_label=[]
            cluster_federated_test_data_for_iid = []
            for idx, el in enumerate(cluster_data[i][1]):
                temp_list_for_image.append(el[0])
                temp_list_for_label.append(el[1])
                if (idx+1)%(num_of_samples[i])==0:
                    cluster_federated_test_data_for_iid.append((np.array(temp_list_for_image, dtype="float32"), np_utils.to_categorical(np.array(temp_list_for_label, dtype="float32"), 47)))
                    temp_list_for_image=[]
                    temp_list_for_label=[]
            list_of_samples_region_test.append(cluster_federated_test_data_for_iid)
        return list_of_samples_region_train, list_of_samples_region_test