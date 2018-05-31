
import numpy as np
import tensorflow as tf

from retro_contest.local import make


import gym_remote.exceptions as gre

#from hashlib import sha1
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import json
import random
import h5py



class bebna_encoder:
    def __init__(self):
        self.info_file = "frame_data.json"
        with open(self.info_file, 'r') as f:
            self.info_dict = json.load(f)
        self.hash_list = []
        for key in self.info_dict:
            self.hash_list.append(key)
        ## Shuffle because they are similarity hashed.
        random.shuffle(self.hash_list)
        #print(self.hash_list)
        self.train_test_split = 1000


        self.config = tf.ConfigProto()
        self.session = tf.Session(config=self.config)

        self.double_height = 224 * 2
        self.double_width = 320 * 2
        self.img_width = 320
        self.img_height = 224
        self.encoder_images = np.zeros((self.double_height, self.img_width, 3))
        plt.ion()
        self.fig, self.ax = plt.subplots(1,1)
        self.image = self.ax.imshow(self.encoder_images[:, :, :], animated=True)
        self.fig.canvas.draw()
        self.make_encoder_decoder()


    def generate_batch(self, batch_type, batch_size):
        ## Load the data file.
        ## Make a list of all the images.
        ## Load a batch worth of images.
        ## Create 24 copies of the image for the puzzle.
        ## Return three things:
        ## 1. An array of the loaded images.
        ## 2. An array of the shuffled images.
        ## 3. An array of the pattern labels for the shuffling.
        permutations = [(4,1,3,2), (4,1,2,3), (4,3,1,2), (4,3,2,1),
                        (4,2,1,3), (4,2,3,1), (1,4,3,2), (1,4,2,3),
                        (1,3,4,2), (1,3,2,4), (1,2,4,3), (1,2,3,4),
                        (3,1,4,2), (3,1,2,4), (3,4,1,2), (3,4,2,1),
                        (3,2,4,1), (3,2,1,4), (2,1,4,3), (2,1,3,4),
                        (2,4,1,3), (2,4,3,1), (2,3,4,1), (2,3,1,4)
                        ]
        num_puzzle_copies = 24
        #end_of_test_set = len(self.hash_list) // self.train_test_split
        end_of_test_set = 100
        batch_images = np.zeros((batch_size, 224, 320, 3))
        batch_puzzle_images = np.zeros((batch_size, 224, 320, 3))
        temp_holder_for_quadrants = np.zeros((4,112,160,3))
        if batch_type == "test":
            current_index = 0 - batch_size
        else:
            current_index = end_of_test_set - batch_size
        while True:
            current_index += batch_size
            if batch_type == "test":
                if current_index > end_of_test_set:
                    current_index = 0 + random.randint(0, batch_size)
            else:
                if current_index > len(self.hash_list) - batch_size:
                    current_index = end_of_test_set

            batch_puzzle_labels = np.zeros((batch_size, num_puzzle_copies))
            ## Get the image hashes for the batch.
            current_hashes = self.hash_list[current_index:current_index+batch_size]
            #print("batch_puzzle_images.shape={}".format(batch_puzzle_images.shape))
            #print("current_hashes={}".format(current_hashes))
            for key, val in enumerate(current_hashes):
                batch_images[key] = np.asarray(Image.open('frames/' + val + '.bmp'))
                ## Set up the puzzle images.
                permutation_index = random.randint(0,23)
                batch_puzzle_labels[key][permutation_index] = 1.0
                current_permutation = permutations[permutation_index]

                ## Split up the current image.
                temp_holder_for_quadrants[current_permutation[0]-1,:,:,:] = batch_images[key,:112, :160,:]
                temp_holder_for_quadrants[current_permutation[1]-1,:,:,:] = batch_images[key,:112, 160:,:]
                temp_holder_for_quadrants[current_permutation[2]-1,:,:,:] = batch_images[key,112:, :160,:]
                temp_holder_for_quadrants[current_permutation[3]-1,:,:,:] = batch_images[key,112:, 160:,:]

                ## Put the quadrants in for prediction.
                batch_puzzle_images[key, :112, :160, :] = temp_holder_for_quadrants[0]
                batch_puzzle_images[key, :112, 160:, :] = temp_holder_for_quadrants[1]
                batch_puzzle_images[key, 112:, :160, :] = temp_holder_for_quadrants[2]
                batch_puzzle_images[key, 112:, 160:, :] = temp_holder_for_quadrants[3]

            #print("batch_puzzle_images.shape={}, sample={}, index={}".format(batch_puzzle_images.shape,batch_images[0][0][0], current_index))

            yield batch_images.astype(np.float32) / 255.0, batch_puzzle_images.astype(np.float32) / 255.0, batch_puzzle_labels


    def animate_data(self, screen_frame, predicted_frame):
        #print(screen_frame[0][0][0], screen_frame[0][0][1], screen_frame[0][0][2])
        #print(np.amax(screen_frame))
        #screen_frame = screen_frame[:,:,::-1]
        #predicted_frame = predicted_frame[:,:,::-1]
        ## Set top half of the encoder images to be the input frame.
        self.encoder_images[:self.img_height,:,:] = screen_frame
        self.encoder_images[self.img_height:, :,:] = predicted_frame
        self.image.set_data(self.encoder_images[:,:,:])
        self.fig.canvas.draw()

    def process_data(self):
        ## We want to run for 5 epochs.
        valid_generator = self.generate_batch('test', 32)
        train_generator = self.generate_batch('train',32)
        for i in range(len(self.hash_list)*50):
            batch_images, batch_puzzle_images, batch_puzzle_labels = train_generator.__next__()
            '''
            self.puzzle_plus_decoder_model.fit(
            {'encoder_input':batch_images,
             'puzzle_encoder_input':batch_puzzle_images,
            },
            {'decoder_output':batch_images,
             'puzzle_output':batch_puzzle_labels},
             32)
            '''
            self.decoder_model.fit({'decoder_encoder_input':batch_images},{'decoder_output':batch_images},32 )

            #self.puzzle_solver_model.fit({'puzzle_encoder_input':batch_puzzle_images},{'puzzle_output':batch_puzzle_labels},32 )
            predicted_frames = self.decoder_model.predict(batch_images)
            #print(predicted_frames)
            self.animate_data(batch_images[0], predicted_frames[0])
        self.decoder_model.save('models/decoder_model.h5')
        self.encoder_model.save('models/encoder_model.h5')
        self.puzzle_solver_model.save('models/puzzle_model.h5')
        self.decoder_eval_model.save('models/decoder_eval_model.h5')

    def mse(self, index):
        ## Use the frame in slot zero of self.observations and the index item
        ## in self.unique_observations.
        return ((self.observations[0] -
               self.unique_observations[index]) ** 2).mean(axis=None)

    def mse_first(self, index_1, index_2):
        return ((self.unique_observations[index_1] -
               self.unique_observations[index_2]) ** 2).mean(axis=None)



    def l2_loss(self, Y_base, Y_pred):
        return tf.nn.l2_loss(Y_base - Y_pred)

    def make_encoder_decoder(self):
        with self.session as sess:
            self.encoder_input = tf.keras.layers.Input(shape=(224, 320, 3,),
                    )
            self.encoder_convolution_0 = tf.keras.layers.Conv2D(16,kernel_size=(4, 4),strides=(4,4), padding='valid')(self.encoder_input)
            self.encoder_convolution_1 = tf.keras.layers.Conv2D(32,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.encoder_convolution_0)
            self.encoder_convolution_2 = tf.keras.layers.Conv2D(64,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.encoder_convolution_1)
            self.encoder_convolution_3 = tf.keras.layers.Conv2D(128,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.encoder_convolution_2)
            self.encoder_flat = tf.keras.layers.Flatten()(
                    self.encoder_convolution_3)
            self.encoder_encoding_layer = tf.keras.layers.Dense(256,
                    )(self.encoder_flat)
            self.encoder_model = tf.keras.models.Model(
                    self.encoder_input, self.encoder_encoding_layer)

            ## Decoder
            ## The encoder segment should be trainable while we are training the
            ## autoencoder so we do not deactivate them here.
            ## Secondary path is based on: https://arxiv.org/pdf/1603.09246.pdf
            ## Secondary path for the decoder. Plan of action:
            ## We are going to slice the image up into units of h:112 w:160
            ## This makes a 4 by 4 grid of the image.
            ## Make a list from zero to 3.
            ## Randomly shuffle that list.

            self.decoder_sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.1,
                    decay=0.1)

            self.puzzle_solver_input = tf.keras.layers.Input(shape=(224, 320, 3,),
                    name='puzzle_encoder_input')
            self.decoder_input = tf.keras.layers.Input(shape=(224, 320, 3,),
                    name='decoder_encoder_input')
            self.puzzle_encoder_layer = self.encoder_model(self.puzzle_solver_input)
            self.decoder_encoder_layer = self.encoder_model(self.decoder_input)

            self.puzzle_solver_layer_1 = tf.keras.layers.Dense(64)(self.puzzle_encoder_layer)
            self.puzzle_solver_layer_2 = tf.keras.layers.Dense(32)(self.puzzle_solver_layer_1)
            self.puzzle_solver_layer_3 = tf.keras.layers.Dense(24)(self.puzzle_solver_layer_2)
            self.puzzle_solver_output = tf.keras.layers.Dense(24, name='puzzle_output')(self.puzzle_solver_layer_3)
            self.puzzle_solver_model = tf.keras.models.Model(self.puzzle_solver_input, self.puzzle_solver_output)
            self.puzzle_solver_model.compile(optimizer='adam',loss={'puzzle_output':'categorical_crossentropy'})


            ## This next line is used if we ever want to be able to run the
            ## decoder by itself. Probably not something we need but it is
            ## there anyways.
            ##self.decoder_input = tf.layers.Input(128, name="decoder_input")
            self.decoder_decoding_layer = tf.keras.layers.Dense(
                    8960)(self.decoder_encoder_layer)
            self.decoder_reshaped = tf.keras.layers.Reshape(
                    target_shape=(7, 10, 128))(self.decoder_decoding_layer)
            self.decoder_deconvolution_1 = tf.keras.layers.Conv2DTranspose(2048,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_reshaped)
            self.decoder_deconvolution_1_2 = tf.keras.layers.Conv2DTranspose(1024,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_reshaped)
            self.decoder_deconvolution_2 = tf.keras.layers.Conv2DTranspose(512,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_deconvolution_1)
            self.decoder_deconvolution_2_2 = tf.keras.layers.Conv2DTranspose(256,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_deconvolution_1_2)
            self.decoder_tower_output = tf.keras.layers.concatenate([self.decoder_deconvolution_2, self.decoder_deconvolution_2_2])
            self.decoder_deconvolution_3 = tf.keras.layers.Conv2DTranspose(128,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_tower_output)
            self.decoder_output = tf.keras.layers.Conv2DTranspose(3,
                    kernel_size=(4,4), strides=(4,4), padding='valid',
                    activation='sigmoid',
                    name="decoder_output")(self.decoder_deconvolution_3)
            #self.decoder_dense_output = tf.keras.layers.Dense(215040, activation='sigmoid')(self.decoder_deconvolution_2)
            #self.decoder_output = tf.keras.layers.Reshape(target_shape=(224, 320, 3),name="decoder_output")(self.decoder_dense_output)

            self.decoder_model = tf.keras.models.Model([self.decoder_input],
                    [self.decoder_output])
            self.decoder_model.compile(optimizer='adam', loss='mse')
            print(self.decoder_model.summary())

            #self.puzzle_plus_decoder_model = tf.keras.models.Model(
            #        inputs=[self.decoder_input, self.puzzle_solver_input],
            #        outputs=[self.decoder_output, self.puzzle_solver_output])
            #self.puzzle_plus_decoder_model.compile(optimizer='sgd',
            #        loss={'decoder_output':'mse', 'puzzle_output':'categorical_crossentropy'},
            #        loss_weights={'decoder_output':0.45, 'puzzle_output':3.0})

            ## Actuator
            ## We make sure here that the encoder is not trainable when we call
            ## the actuator.
            self.encoder_convolution_0.trainable = False
            self.encoder_convolution_1.trainable = False
            self.encoder_convolution_2.trainable = False
            self.encoder_convolution_3.trainable = False
            self.encoder_encoding_layer.trainable = False

            ## For looking at decoder output.
            self.decoder_decoding_layer.trainable = False
            self.decoder_reshaped.trainable = False
            self.decoder_deconvolution_1.trainable = False
            self.decoder_deconvolution_2.trainable = False
            self.decoder_deconvolution_3.trainable = False
            self.decoder_output.trainable = False
            self.decoder_eval_model = tf.keras.models.Model([self.decoder_input],
                    [self.decoder_output])
            self.decoder_eval_model.compile(optimizer=self.decoder_sgd, loss='mse')
            ##print(self.decoder_model.summary())
            ## Also make and compile the encoder model here so it can be used
            ## in a frozen manner.
            #self.encoder_model = tf.keras.models.Model(
            #        inputs=[self.encoder_input],
            #        outputs=[self.encoder_encoding_layer])

            print(self.encoder_model.summary())

            #self.encoder_sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0,
            #        decay=0.0, nesterov=True)
            self.encoder_model.compile(optimizer='adam', loss='mse')











def main():
    ## do stuff here
    trainer = bebna_encoder()
    trainer.process_data()

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
