# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# ai libraries
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

def gpu_session():
    """[Initiates the GPU session. External code integrated.]
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 6GB of memory on the first GPU
        # In this way we avoid memory overflow
        # TODO: Check how to get a percentage of the free GPU memory, instead of allocating a fixed number (nvidia-smi?)
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(
                                                                        memory_limit=3 * 1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
def data_management(epochs,save_dir, model_name,model_optimizer,batch_size,seed):

    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            pass

    # Create the datagenerators
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='mirror',
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    val_datagen  = ImageDataGenerator(rescale = 1./255)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    # ------------------------------------------------------------------------------------------------------------------
    # PREPROCESSING DATA
    # ------------------------------------------------------------------------------------------------------------------
    # Read the labels
    train_set_url = 'C:/Users/Javi/Desktop/Master/Kaggle_ballenas/train_images'
    # test_set_url = '../input/happy-whale-and-dolphin/test_images'
    tran_labels_url = 'C:/Users/Javi/Desktop/Master/Kaggle_ballenas/train.csv'

    df = pd.read_csv(tran_labels_url, header=0)
    df.drop(['individual_id'], inplace=True, axis=1)

    # Check if the data contain any unapproppriate inputs. If so, correct them
    if df.isnull().values.any() == True:
        df.isnull().sum()  # count the null values
        df.fillna(0, inplace=True)  # to replance null values i.e., "NaN" with "0"
        print('Data is managed')
    else:
        print('All data is valid')

    # Obtain all the metadata
    classes_old = df['species'].unique()
    names_old = df['image']
    n_whales_old = len(names_old)
    # print(n_whales_old)
    # Prepare the 'image column to have the absolute path included per image'
    for i in range(len(names_old)):
        df.at[i, 'image'] = train_set_url + '/' + names_old[i]

    # print(df.shape[0])
    # Decimate under represented classes
    # -----------------------------------------------------------
    # TODO
    class_count_df = df['species'].value_counts()
    list = []
    for i in range(len(classes_old)):
        list.append(class_count_df[i])
        if list[i] <= 1000:
            check = classes_old.item(i)

            df = df.drop(df[df['species'] == check].sample(frac=1).index)
    # Obtain all new the metadata
    classes = df['species'].unique()
    names = df['image']
    n_whales = len(names)
    print('Se usarán un total de: ', len(classes))

    #-----------------------------------------------------------

    # CREATE THE BATCHES
    train_ratio = 0.75
    val_ratio = 0.20
    test_ratio = 0.05
    class_mode = 'categorical'
    train_size = round(train_ratio*n_whales)
    val_size = round(val_ratio*n_whales)
    test_size = round(test_ratio*n_whales)

    data = []
    # Split the data into Train, Validation and Test
    # First Shuffle
    df = df.sample(frac = 1)
    # Then split
    lim_bot = 0
    lim_top = train_size
    train_set = df.iloc[lim_bot:lim_top,:]
    # Reduce and compensate the number of elements per class
    class_count = train_set['species'].value_counts()
    list = []
    for i in range(len(classes)):
        list.append(class_count[i])
        if list[i] >= 1000:
            check = classes.item(i)
            train_set = train_set.drop(train_set[train_set['species'] == check].sample(frac=0.8).index)

    lim_bot = train_size
    lim_top = train_size + val_size
    val_set = df.iloc[lim_bot:lim_top, :]
    # Reduce and compensate the number of elements per class
    class_count = val_set['species'].value_counts()
    list = []
    for i in range(len(classes)):
        list.append(class_count[i])
        if list[i] >= 1000:
            check = classes.item(i)
            val_set = val_set.drop(val_set[val_set['species'] == check].sample(frac=0.5).index)

    lim_bot = train_size + val_size
    lim_top = train_size + val_size + test_size
    test_set = df.iloc[lim_bot:lim_top, :]

    # Create the dataset generators
    train_generator = train_datagen.flow_from_dataframe (train_set, x_col = 'image', y_col = 'species',target_size = (224,224), class_mode = class_mode, batch_size = batch_size, shuffle = True)
    val_generator = val_datagen.flow_from_dataframe (val_set, x_col = 'image', y_col = 'species',target_size = (224,224), class_mode = class_mode, batch_size = batch_size, shuffle = False)
    test_generator = test_datagen.flow_from_dataframe (test_set, x_col = 'image', y_col = 'species',target_size = (224,224), class_mode = class_mode, batch_size = batch_size, shuffle = False)
    return train_generator, val_generator, test_generator, classes, test_set
def model_design_and_compile(classes):
    # Create the model
    model = MobileNetV2(input_shape=(224, 224, 3),
                        alpha=0.35,
                        include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        pooling=None)
    for layer in model.layers:
        layer.trainable = True

    # aux0 = model.output
    # aux1 = GlobalAveragePooling2D()(aux0)
    # aux2 = Dense(512, activation='relu')(aux1)
    # aux3 = Dense(256, activation='relu')(aux2)
    # aux4 = Dense(128, activation='relu')(aux3)
    # aux5 = Dense(64, activation='relu')(aux4)
    # aux6 = Dense(len(classes), activation='softmax')(aux5)
    # new_model = Model(model.input, aux6)

    print("Las clases son:", len(classes))
    aux0 = model.output
    aux1 = GlobalAveragePooling2D()(aux0)
    aux2 = Dense(512, activation='relu')(aux1)
    drop2 = Dropout(0.7)(aux2)
    aux3 = Dense(256, activation='relu')(drop2)
    drop3 = Dropout(0.6)(aux3)
    aux4 = Dense(128, activation='relu')(drop3)
    drop4 = Dropout(0.5)(aux4)
    aux5 = Dense(64, activation='relu')(drop4)
    drop5 = Dropout(0.3)(aux5)
    aux6 = Dense(len(classes), activation='softmax')(drop5)
    new_model = Model(model.input, aux6)

    new_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      metrics=['acc', 'MeanSquaredError',
                               'AUC'])
    return new_model
def fit(new_model,epochs,train_generator,val_generator,save_dir, model_name,model_optimizer,batch_size):
    gpu_name = tf.test.gpu_device_name()
    print('The GPU is' + gpu_name)
    #if device_name != '/device:GPU:0':
    #  raise SystemError('GPU device not found')
    #print('Found GPU at: {}'.format(device_name))
    gpu_session()

    # checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1,
    #                              save_best_only=True,
    #                              mode='max')  # graba sólo los que mejoran en validación
    # callbacks_list = [checkpoint]
    acum_tr_acc = []
    acum_val_acc = []
    hist = []
    with tf.device(gpu_name):
        print('in session')
        #for e in range(epochs):
        hist = new_model.fit(train_generator, steps_per_epoch=round(train_generator.samples/train_generator.batch_size), epochs = epochs,  validation_data = val_generator,
                    validation_steps=round(val_generator.samples/val_generator.batch_size), verbose = 1)#, callbacks = callbacks_list)#, steps_per_epoch=len(train_generator),validation_steps=len(val_generator))
            #acum_tr_acc.append(hist.history['acc'][0])
            #acum_val_acc.append(hist.history['val_acc'][0])
            #if len(acum_tr_acc) > 0:
            #    model_save(save_dir, new_model, model_name,model_optimizer,batch_size,acum_val_acc)
    return hist
def plot_roc_curve(tpr, fpr, scatter=True, ax=None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).

    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize=(5, 5))
        ax = plt.axes()

    if scatter:
        sns.scatterplot(x=fpr, y=tpr, ax=ax)
    sns.lineplot(x=fpr, y=tpr, ax=ax)
    sns.lineplot(x=[0, 1], y=[0, 1], color='green', ax=ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list
def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''

    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculates tpr and fpr
    tpr = TP / (TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate

    return tpr, fpr
def roc_curve_ovr(classes,y_test,y_proba):
    # Plots the Probability Distributions and the ROC Curves One vs Rest
    plt.figure(figsize=(26, 16))
    bins = [i / 20 for i in range(20)] + [1]
    roc_auc_ovr = {}
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]
        print(str(c))
        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame(np.zeros(shape=(2042, 2)), columns=['class', 'prob'])
        #df_aux = X_test.copy()
        df_aux['class'] = [1 if y == i else 0 for y in y_test]
        df_aux['prob'] = y_proba[:, i]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the class and the rest
        if i < 7:
            ax = plt.subplot(4, 7, i + 1)
            sns.histplot(x="prob", data=df_aux, hue='class', color='b', ax=ax, bins=bins)
            ax.set_title(c)
            ax.legend([f"Class: {c}", "Rest"])
            ax.set_xlabel(f"P(x = {c})")
        elif 7 <= i < 14:
            ax = plt.subplot(4, 7, i + 8)
            sns.histplot(x="prob", data=df_aux, hue='class', color='b', ax=ax, bins=bins)
            ax.set_title(c)
            ax.legend([f"Class: {c}", "Rest"])
            ax.set_xlabel(f"P(x = {c})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        if i < 7:
            ax_bottom = plt.subplot(4, 7, i + 8)
            tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
            plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
            ax_bottom.set_title("ROC Curve OvR")
        elif 7 <= i < 14:
            ax_bottom = plt.subplot(4, 7, i + 15)
            tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
            plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
            ax_bottom.set_title("ROC Curve OvR")

        # Calculates the ROC AUC OvR
        roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])
    plt.tight_layout()
def reporting_save(report,modelHistory, test_info, conf_mat, model_name, model_opt, batch_size):
    # Generate the file names
    x = datetime.datetime.now()
    year = str(x.year)
    month = str(x.month)
    day = str(x.day)
    summary_name = "_summary"
    # pb_model_name = (year + "_" + month + '_'+ day + '_' + str(self.model_name) + '_'
    #     + str(self.model_opt) + '_' + type_decay + '_' + str(self.batch_size) + "_model")
    model_name = (year + "_" + month + '_' + day + '_' + str(model_name) + '_'
                  + str(model_opt) + str(
                batch_size) + '_model.json')
    history_name = (year + "_" + month + '_' + day + '_' + str(model_name) + '_'
                    + str(model_opt) +  '_' + str(
                batch_size) + "_hist")

    # Create and save the summary file
    hist_df = pd.DataFrame(modelHistory.history)

    sum_df = [[model_name, batch_size,
               model_opt, modelHistory.history['loss'][-1],
               modelHistory.history['val_loss'][-1], test_info[0],
               modelHistory.history['acc'][-1],
               modelHistory.history['val_acc'][-1], test_info[1]]]

    conf_array = np.array2string(conf_mat)

    # Create and save summary of best records
    if os.path.isfile(summary_name):
        with open(summary_name, mode='a') as f:
            for item in sum_df:
                f.write("%s\n" % item)
                f.write('%s\n' % report)
                f.write('%s\n\n' % conf_array)
    else:
        with open(summary_name, mode='w') as f:
            f.write(str(['model_name', 'percentage', 'model mode', 'batch_size', 'opt', 'lr', 'type_decay',
                         'loss', 'val_loss', 'test_loss', 'acc', 'val_acc', 'test_acc']))
            f.write('\n')
            for item in sum_df:
                f.write('%s\n' % item)
                f.write('%s\n' % report)
                f.write('%s\n\n' % conf_array)
                # f.write('%s\n\n' %sensitivity.result().numpy())

    # Create and save the historical of all epochs (train_loss, val_loss, train_acc,val_acc)
    # into a csv
    with open(history_name, mode='w') as f:
        hist_df.to_csv(f)
def model_save(save_dir,model, name, model_opt, batch_size, acum_val_acc):

    """[Is used for saving a summary of best test records, the compiled model,
    a history of train values and the weights for the compiled model]

    Args:
        save_dir ([string]): [directory for saving the data]
        model ([type]): [description]
        modelHistory ([History Object]): [is a record of training loss values and
                                    metrics values at successive epochs, as well as
                                    validation loss values and validation metrics
                                    values]
        test_info ([lit]): [List of train loss and accuracy values]
        type_decay ([string]): [[Describes the learning rate decay used for fit method]]
    """
    # Generate the file names
    x = datetime.datetime.now()
    year = str(x.year)
    month = str(x.month)
    day = str(x.day)
    summary_name = "_summary"
    # pb_model_name = (year + "_" + month + '_'+ day + '_' + str(self.model_name) + '_'
    #     + str(self.model_opt) + '_' + type_decay + '_' + str(self.batch_size) + "_model")
    model_name = (year + "_" + month + '_' + day + '_' + str(name) + '_'
                  + str(model_opt) + str(
                batch_size) + '_val_acc_' + '_model.json')
    weights_name = (year + "_" + month + '_' + day + '_' + str(name) + '_'
                    + str(model_opt) + '_' + str(
                batch_size) + '_val_acc_' + '_weights.h5')
    # ----------------------------------------------------------------------------------------------------------------------------
    # SAVE THE MODEL
    # ----------------------------------------------------------------------------------------------------------------------------
    # Create and save the model into .pb format
    # dir_pb = os.getcwd() + '/' + pb_model_name
    # tf.saved_model.save(model,dir_pb)
    # Create and save the model into a JSON file
    model_json = model.to_json()
    with open(save_dir + '/' + model_name, 'w') as json_file:
        json_file.write(model_json)

    # Create and save the weights into a h5 file
    model.save_weights(save_dir + '/' + weights_name)
    print('Model trained saved at %s' % save_dir + '/')

def main():
    save_dir = 'C:/Users/Javi/Desktop/Master/Kaggle_ballenas/Results'
    model_name = 'mobilenet_whales'
    model_optimizer = 'Adam'
    epochs = 100
    batch_size = 32
    # Create the seed
    seed = np.random.seed(1)

    # ------------------------------------------------------------------------------------------------------------------
    # DATA MANAGEMENT
    # ------------------------------------------------------------------------------------------------------------------
    train_generator, val_generator, test_generator, classes, test_set = data_management(epochs,save_dir, model_name, model_optimizer, batch_size, seed)
    x_train, y_train = train_generator.next()
    x_val, y_val = val_generator.next()
    x_test, y_test = test_generator.next()
    print('Data management done')

    # ------------------------------------------------------------------------------------------------------------------
    # MODEL DESIGN AND COMPILE
    # ------------------------------------------------------------------------------------------------------------------
    new_model = model_design_and_compile(classes)
    print('Compiled model')

    # ------------------------------------------------------------------------------------------------------------------
    # FIT THE MODEL
    # ------------------------------------------------------------------------------------------------------------------
    hist = fit(new_model, epochs, train_generator, val_generator, save_dir, model_name, model_optimizer, batch_size)

    # ------------------------------------------------------------------------------------------------------------------
    # EVALUATE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------
    # gpu_session()
    # Open the model and load the info
    # json_file = open('C:/Users/Javi/Desktop/Master/Kaggle_ballenas/Results/2022_4_6_mobilenet_whales_Adam32_val_acc_0.3910590410232544_model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # new_model = tf.keras.models.model_from_json(loaded_model_json)

    # Load weights into new model
    # new_model.load_weights('C:/Users/Javi/Desktop/Master/Kaggle_ballenas/Results/2022_4_6_mobilenet_whales_Adam_32_val_acc_0.3910590410232544_weights.h5')
    #
    # new_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    #                   metrics=['acc'])

    scores = new_model.evaluate(test_generator, verbose=1)
    test_info = [scores[0], scores[1]]
    print('Test loss: ', scores[0])
    print('Test accuracy: ', scores[1])

    y_pred = new_model.predict(test_generator)
    y_real = test_generator.classes
    preds = np.argmax(y_pred, axis=1)
    class_labels = test_set['species'].unique()

    report = classification_report(test_generator.classes, preds, target_names=class_labels)
    conf_mat = confusion_matrix(test_generator.classes, preds)

    roc_curve_ovr(class_labels, y_real, y_pred)
    print('done')

    # save plot to file
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    os.chdir(save_dir)
    print(model_name)

    filename = 'roc-auc curve_' + model_name + '_' + str(model_optimizer) + '_' + str(batch_size)
    plt.savefig(filename + '_plot.png')
    plt.close()
    print('Plot saved at %s' % os.getcwd())

    # Plot the different metrics and results for evaluating the model performance
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['acc'], 'r')
    plt.plot(hist.history['val_acc'], 'g')
    plt.xticks(np.arange(0, epochs, int(epochs)))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train', 'validation'])

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], 'r')
    plt.plot(hist.history['val_loss'], 'g')
    plt.xticks(np.arange(0, epochs, int(epochs)))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train', 'validation'])

    # save plot to file
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    os.chdir(save_dir)
    print(model_name)

    filename = 'train_val_accu_and_loss_' + model_name + '_' + str(model_optimizer) + '_' + str(batch_size)
    plt.savefig(filename + '_plot.png')
    plt.close()
    print('Plot saved at %s' % os.getcwd())

    reporting_save(report, hist, test_info, conf_mat, model_name, model_optimizer, batch_size)
    acum_val_acc = 0
    model_save(save_dir, new_model, model_name, model_optimizer, batch_size, acum_val_acc)

if __name__ == "__main__":
    main()