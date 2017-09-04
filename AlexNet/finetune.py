import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator


image_dir_root_pep = r"D:\ETH\RA\Aqua_Peptide_Extraction\mzxml_decoy\Black_BG_227\guot_PC1_170125_CPP2"
peptide_dict_enumerated = dict(enumerate(os.listdir(image_dir_root_pep)))

print(peptide_dict_enumerated)

# Path to the textfiles for the trainings and validation set
train_file = 'train.txt'
val_file = 'val.txt'

# Learning params
learning_rate = 0.001 
num_epochs = 50
batch_size = 16

# Network params
dropout_rate = 0.5
num_classes = 16
#train_layers = ['fc8', 'fc7', 'fc6']
train_layers = ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = r"/tmp/finetune_alexnet/tensorboard"
checkpoint_path = r"/tmp/finetune_alexnet/checkpoints"
confusion_matrices = r"/tmp/confusion_matrices"
if not os.path.exists(filewriter_path):
    os.makedirs(filewriter_path)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.exists(confusion_matrices):
    os.makedirs(confusion_matrices)

with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)


# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(score, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#with tf.name_scope("confusion_matrix_tf"):
#    confusion_matrix_tf = tf.confusion_matrix(tf.argmax(score, 1), tf.argmax(y, 1))
    
precision_tf, precision_op = tf.metrics.precision(tf.argmax(y, 1), tf.argmax(score, 1))
recall_tf, recall_op = tf.metrics.recall(tf.argmax(y, 1), tf.argmax(score, 1))
    
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

mat_count = 0

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))
    training_loss = []
    validation_loss = []
    training_time_array = []
    validation_time_array = []
    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            _,train_loss = sess.run([train_op,loss], feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})
            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s,train_loss = sess.run([merged_summary,loss], feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})
                #print("Training Loss = {}".format(train_loss))
                #training_loss.append(train_loss)
                #training_time_array.append(datetime.now())
                #tf.summary.scalar('train_loss', train_loss)
                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            plt.gcf().clear()   
            img_batch, label_batch = sess.run(next_batch)
            acc,val_loss = sess.run([accuracy,loss], feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            '''
            if _ % 50 == 0:
	            cm = confusion_matrix_tf.eval(feed_dict={x: img_batch,
	                                                y: label_batch,
	                                                keep_prob: 1.})
	            
	            precision = sess.run(precision_op, feed_dict={x: img_batch,
	                                                y: label_batch,
	                                                keep_prob: 1.})
	            recall = sess.run(recall_op, feed_dict={x: img_batch,
	                                                y: label_batch,
	                                                keep_prob: 1.})
	            f1_score = 2 * ((precision*recall)/(precision+recall))
	            plt.gcf().clear()
	            plt.figure()
	            fig, ax = plt.subplots(figsize=(25,25))     
	            #heatmap_plot = sns.heatmap(cm,annot=True,cbar=True,fmt="d",linewidths=1,xticklabels=list(peptide_dict_enumerated.values()),yticklabels=list(peptide_dict_enumerated.values()),ax=ax)
	            heatmap_plot = sns.heatmap(cm,annot=True,cbar=True,fmt="d",linewidths=1,ax=ax)
	            plt_fig = heatmap_plot.get_figure()
	            plt_fig.savefig(os.path.join(confusion_matrices,"{}Mat_P{}R{}E{}F{}.png".format(mat_count,precision,recall,epoch+1,f1_score)))
	            plt.clf()
	            mat_count = mat_count + 1
	            print("Precision = {}".format(precision))
	            print("Recall = {}".format(recall))
	            print("F1 Score = {}".format(f1_score))
	            print("Validation Loss = {}".format(val_loss))
	            validation_loss.append(val_loss)
	            validation_time_array.append(datetime.now())
            '''
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
    model_name = os.path.join(checkpoint_path, "ModelL{}D{}B{}".format(learning_rate,dropout_rate,batch_size))
    saver.save(sess,model_name)
'''
with open("Training_Time.txt", "wb") as fp:
    pickle.dump(training_time_array, fp)

with open("Training_Loss.txt", "wb") as fp:   
    pickle.dump(training_loss, fp)

with open("Validation_Time.txt", "wb") as fp:
    pickle.dump(validation_time_array, fp)

with open("Validation_Loss.txt", "wb") as fp:   
    pickle.dump(validation_loss, fp)

plt.gcf().clear()

plt.figure()
plt.plot(training_time_array, training_loss)
plt.title('Training Loss')
plt.xlabel("Time")
plt.ylabel("Loss")
plt.savefig('training_loss_graph.png')

plt.gcf().clear()

plt.figure()
plt.plot(validation_time_array, validation_loss)
plt.title('Validation Loss')
plt.xlabel("Time")
plt.ylabel("Loss")
plt.savefig('validation_loss_graph.png')
'''