from utils.header import *
from packages.dataset.dataset import JointKDDataset
from packages.engine.derivative_nw import TeacherNetwork, StudentNetwork
from utils.c_reliability import CReliability
from utils.helper_function import *

def main():
    start=datetime.now()
    # get configure
    g_config = GeneralConfigure()
    # prepare dataset
    dataset = JointKDDataset()
    cluster_data = dataset.preProcessData()
    list_of_samples_region_train, list_of_samples_region_test = dataset.assignDataForClients(cluster_data)
    
    # get student network
    student_model = StudentNetwork()
    student  = keras.models.Model(inputs  = student_model.model.input,                      
                                 outputs = student_model.model.get_layer('logits').output)
    student.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
    # define distilation loss function
    distillation_loss_fn = keras.losses.KLDivergence()
    # create teacher model
    train_able = False
    if train_able == True:
        teacher_params = []
        teacher = []
        for region in range(g_config.regions):
            teacher.append(TeacherNetwork(list_of_samples_region_train[region+1],
                                        list_of_samples_region_test[region+1],
                                        list_of_samples_region_train[0],
                                        list_of_samples_region_test[0],
                                        student,
                                        distillation_loss_fn,
                                        g_config.model_path, region, dataset.num_classes))
            teacher_param, _ = teacher[region].regionalAggregation()
            teacher_params.append(teacher_param)
    else:
        pass
    # C-Reliability 
    c_reliability = CReliability(list_of_samples_region_train[0][0], dataset.num_classes)
    beta = c_reliability.weightedClass()

    dataset_server = processDataServer(list_of_samples_region_train[0][0])
    for epoch in range(g_config.distil_epochs):
        print("Start of epoch %d" % (epoch,))
        for step, batch_train in enumerate(dataset_server):
            loss = 0
            x_batch_tf, y_batch_tf = batch_train
            x_batch = x_batch_tf.numpy()
            y_batch = y_batch_tf.numpy()
            # print(len(x_batch))
            # print(len(y_batch))
            for region in range(g_config.regions):
                teacher_model = tf.keras.models.load_model(os.path.join(c_reliability.model_path,f"teacher_region_{region}.h5"),compile=False)
                logit_teacher_model = keras.models.Model(inputs  = teacher_model.input, outputs = teacher_model.get_layer('logits').output)
                logit_teacher_model.compile(optimizer='adam',
                                            loss='sparse_categorical_crossentropy',
                                            metrics=['sparse_categorical_accuracy'])
                logits_predict = logit_teacher_model(x_batch, training=False)
                softmax_predict = softmax(logits_predict)
                rounded_predict = np.argmax(softmax_predict, axis = 1)
                pseudo_dataset = list(zip(x_batch, rounded_predict, y_batch))

                aligned_data_teacher, aligned_label_teacher = dataAlignment(pseudo_dataset, dataset.num_classes)
                teacher_predictions = {}
                for label in range(dataset.num_classes):
                    # Calculate individual loss by 
                    # Forward pass of teacher
                    teacher_predictions[label] = logit_teacher_model(aligned_data_teacher[label].reshape(-1,28,28,1), training=False)
                with tf.GradientTape() as tape:
                    # Calculate each label-driven loss
                    # Forward pass of student
                    for label in range(dataset.num_classes):
                        student_predictions = student(aligned_data_teacher[label].reshape(-1,28,28,1), training=True)
                        # Compute losses at $label$ round
                        student_loss = g_config.student_loss_fn(aligned_label_teacher[label], student_predictions)
                        distillation_loss = beta[region][label] * distillation_loss_fn(tf.nn.softmax(teacher_predictions[label] / 20, axis=1),
                                                                    tf.nn.softmax(student_predictions        / 20, axis=1))
                        if label == 0: 
                            concat_student_predictions = student_predictions
                        else:
                            concat_student_predictions = tf.concat([concat_student_predictions,student_predictions], 0)
                        loss += g_config.alpha * student_loss + (1 - g_config.alpha) * distillation_loss
            # Compute gradients
            trainable_vars = student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            g_config.optimizer.apply_gradients(zip(gradients, trainable_vars))       

            if step % 2 == 0:
                print(f"step {step}: studentloss = {student_loss}, distillation loss = {loss}, accuracy = {g_config.accuracy_metric.result()}")
                student.evaluate(x = list_of_samples_region_test[0][0], y = list_of_samples_region_test[0][1])

    stop = datetime.now()
    print('Total Time: ', stop - start)


if __name__ == "__main__":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    main()