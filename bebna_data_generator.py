
import numpy as np
import tensorflow as tf

from retro_contest.local import make


import gym_remote.exceptions as gre

#from hashlib import sha1
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import json



class bebna:
    def __init__(self, initial_random_actions=256, num_actions=12,
            num_frames_before_show=100):
        self.initial_random_actions = initial_random_actions
        self.num_frames_before_show = num_frames_before_show
        self.current_frame = 0
        self.num_actions = num_actions


        self.b_action = 0
        self.a_action = 1
        self.mode_action = 2
        self.start_action = 3
        self.up_action = 4
        self.down_action = 5
        self.left_action = 6
        self.right_action = 7
        self.c_action = 8
        self.y_action = 9
        self.x_action = 10
        self.z_action = 11

        self.empty_train = 0
        self.left_train = 1
        self.right_train = 2
        self.left_down_train = 3
        self.right_down_train = 4
        self.down_train = 5
        self.down_b_train = 6
        self.left_b_train = 7
        self.right_b_train = 8
        self.b_train = 9
        '''
        self.opposites = {
                self.left_train:[self.right_train,
                    self.right_down_train, self.right_b_train],
                self.right_train:[self.left_train,
                    self.left_down_train, self.right_b_train],
                self.left_down_train:[self.right_train,
                    self.right_down_train, self.right_b_train],
                self.right_down_train:[self.left_train,
                    self.left_down_train, self.left_b_train],
                self.down_train:[self.left_train, self.right_train,
                    self.left_b_train, self.right_b_train],
                self.down_b_train:[self.left_train, self.right_train,
                    self.left_b_train, self.right_b_train],
                self.left_b_train:[self.right_train, self.right_b_train,
                    self.down_b_train, self.right_down_train],
                self.right_b_train:[self.left_train, self.left_b_train,
                    self.down_b_train, self.left_down_train],
                self.b_train:[self.left_train, self.right_train,
                    self.left_down_train, self.right_down_train,
                    self.down_train, self.down_b_train, self.left_b_train,
                    self.right_b_train],
                self.empty_train:[self.right_train,
                    self.right_down_train,
                    self.down_train, self.down_b_train,
                    self.right_b_train,
                    self.b_train]
        }
        '''
        self.opposites = {
                self.left_train:[self.right_train,
                    self.right_down_train, self.right_b_train],
                self.right_train:[self.left_train,
                    self.left_down_train, self.right_b_train],
                self.left_down_train:[self.right_train,self.right_b_train],
                self.right_down_train:[self.left_train,self.left_b_train],
                self.down_train:[self.left_train, self.right_train,
                    self.left_b_train, self.right_b_train],
                self.down_b_train:[self.left_train, self.right_train,
                    self.left_b_train, self.right_b_train],
                self.left_b_train:[self.right_train, self.right_b_train, self.right_down_train],
                self.right_b_train:[self.left_train, self.left_b_train, self.left_down_train],
                self.b_train:[self.left_train, self.right_train,
                    self.left_down_train, self.right_down_train,
                    self.down_b_train, self.left_b_train,
                    self.right_b_train],
                self.empty_train:[self.right_train,
                    self.right_down_train,
                    self.down_train, self.down_b_train,
                    self.right_b_train,
                    self.b_train]
        }



    def setup(self, env, observation_hashes={}):
        self.env = env

        self.last_action = self.env.action_space.sample()
        self.last_hash = None
        self.current_random_actions = 0

        self.observations = np.zeros((16, 224, 320, 3))
        self.num_observations = 0
        self.unique_observations = np.zeros((16,224,320, 3))
        ## How different the unique observations are to each other.
        ## We want high numbers here, rather than low ones.
        self.similarity = np.zeros((16, 16))
        self.uniqueness = np.zeros(16)
        self.uniqueness_check = np.zeros(16)
        self.observation_hashes = observation_hashes
        self.config = tf.ConfigProto()
        self.session = tf.Session(config=self.config)
        self.encoder_model = None
        self.decoder_model = None
        self.actuator_model = None
        self.make_encoder_decoder()
        ## Stuff for showing how the encoder is doing.
        self.img_height = 224
        self.img_width = 320
        self.double_height = 448
        #self.encoder_images = np.zeros((self.double_height, self.img_width, 3))
        #plt.ion()
        #self.fig, self.ax = plt.subplots(1,1)
        #self.image = self.ax.imshow(self.encoder_images[:, :, :], animated=True)
        #self.fig.canvas.draw()


    def animate_data(self, screen_frame, predicted_frame):
        #print(screen_frame[0][0][0], screen_frame[0][0][1], screen_frame[0][0][2])
        #print(np.amax(screen_frame))
        #screen_frame = screen_frame[:,:,::-1]
        predicted_frame = predicted_frame[:,:,::-1]
        ## Set top half of the encoder images to be the input frame.
        self.encoder_images[:self.img_height,:,:] = screen_frame
        self.encoder_images[self.img_height:, :,:] = screen_frame#predicted_frame
        self.image.set_data(self.encoder_images[:,:,:])
        #self.fig.canvas.draw()

    def save_image(self, data, hash, info, f):
        #self.animate_data(data, data)
        data = data * 255.0
        data = data.astype('uint8')

        #print(data.dtype)
        img_to_save = Image.fromarray(data, mode='RGB')
        img_to_save.save('frames/' + str(hash) + '.bmp')
        f[str(hash)] = info


    def convert_to_train_space(self, data):
        ## Data is 12 units long.
        ## Map from base action space to reasonable action space.
        new_data = [0]*10
        if data[self.down_action] == 1:
            if data[self.b_action] == 1:
                new_data[self.down_b_train] = 1
            elif data[self.left_action] == 1:
                new_data[self.left_down_train] = 1
            elif data[self.right_action] == 1:
                new_data[self.right_down_train] = 1
            else:
                new_data[self.down_train] = 1
        elif data[self.b_action] == 1:
            if data[self.right_action] == 1:
                new_data[self.right_b_train] = 1
            elif data[self.left_action] == 1:
                new_data[self.left_b_train] = 1
            else:
                new_data[self.b_train] = 1
        elif data[self.left_action] == 1:
            new_data[self.left_train] = 1
        elif data[self.right_action] == 1:
            new_data[self.right_train] = 1
        else:
            new_data[self.empty_train] = 1
        return new_data

    def convert_to_action_space(self, data):
        ## Data is one integer.
        new_data = [0]*12
        if data == self.empty_train:
            pass
        if data == self.left_train:
            new_data[self.left_action] = True
        if data == self.right_train:
            new_data[self.right_action] = True
        if data == self.left_down_train:
            new_data[self.left_action] = True
            new_data[self.down_action] = True
        if data == self.right_down_train:
            new_data[self.right_action] = True
            new_data[self.down_action] = True
        if data == self.down_train:
            new_data[self.down_action] = True
        if data == self.down_b_train:
            new_data[self.down_action] = True
            new_data[self.b_action] = True
        if data == self.b_train:
            new_data[self.b_action] = True
        return new_data

    def process_data(self, obs, info={}, f=None):
        #input('...')
        temp_hash = imagehash.whash(Image.fromarray(obs, mode='RGB'))
        if self.current_random_actions < self.initial_random_actions:
            output_data = self.env.action_space.sample()
            self.current_random_actions += 1
        else:

            if np.array_equal(self.last_action, np.array([1, 0, 0, 0, 0, 0, 0,1, 0, 0, 0, 0])):
                output_data = np.array([0, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0])
            else:
                output_data = np.array([1, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0])


        ## Add this observation to the set of ones seen.
        ## Update the network, save the output from this, and return the new set
        ## of instructions.
        self.add_observation(obs, temp_hash, info, f)
        return output_data


    def add_observation(self, obs, temp_hash, info, f):
        ## Update the actuator model given the previous output and the current
        ## screen.
        self.new_observation_check(temp_hash, obs, info, f)


    def new_observation_check(self, current_hash, obs, info, f):
        ## Only check if we have seen 16 observations.
        found_similar = False
        for temp_hash in self.observation_hashes:
            #print(temp_hash - current_hash)
            if temp_hash - current_hash < 18:
                found_similar = True
        if not found_similar:
            self.observation_hashes[current_hash] = 0
            self.save_image(obs, current_hash, info, f)

    def make_encoder_decoder(self):
        with self.session as sess:
            try:
                #self.encoder_model = tf.keras.models.load_model('models/encoder_model')
                self.encoder_frozen_model = tf.keras.models.load_model('models/encoder_model.h5')
                for layer in self.encoder_frozen_model.layers:
                    layer.trainable=False

                #self.decoder_model = tf.keras.models.load_model('models/decoder_model')
                self.decoder_frozen_model = tf.keras.models.load_model('models/decoder_model.h5')
                for layer in self.decoder_frozen_model.layers:
                    layer.trainable=False

                #self.puzzle_solver_model = tf.keras.models.load_model('models/puzzle_model.h5')
                #self.decoder_eval_model = tf.keras.models.load_model('models/decoder_eval_model.h5')

            except:
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
                self.decoder_deconvolution_1 = tf.keras.layers.Conv2DTranspose(256,
                        kernel_size=(2,2), strides=(2,2), padding='valid',
                        activation='relu')(self.decoder_reshaped)
                self.decoder_deconvolution_1_2 = tf.keras.layers.Conv2DTranspose(256,
                        kernel_size=(2,2), strides=(2,2), padding='valid',
                        activation='relu')(self.decoder_reshaped)
                self.decoder_deconvolution_2 = tf.keras.layers.Conv2DTranspose(128,
                        kernel_size=(2,2), strides=(2,2), padding='valid',
                        activation='relu')(self.decoder_deconvolution_1)
                self.decoder_deconvolution_2_2 = tf.keras.layers.Conv2DTranspose(128,
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


            ## Actuator code.
            self.actuator_input = tf.keras.layers.Input(shape=(224, 320, 3,),
                    name='actuator_input')
            self.actuator_encoded = self.encoder_model(self.actuator_input)
            self.actuator_dense_1 = tf.keras.layers.Dense(64)(
                    self.actuator_encoded)
            self.actuator_dense_2 = tf.keras.layers.Dense(32)(
                    self.actuator_dense_1)
            self.actuator_dense_3 = tf.keras.layers.Dense(16)(
                    self.actuator_dense_2)
            self.actuator_output = tf.keras.layers.Dense(10,
                    name="actuator_output",
                    activation="softmax")(self.actuator_dense_3)
            self.actuator_model = tf.keras.models.Model([self.actuator_input],
                    [self.actuator_output])

            self.actuator_model.compile(optimizer='adam',
                    loss='categorical_crossentropy')

            #tf.initialize_all_variables().run()
            tf.global_variables_initializer().run()









def main():

    envs = [('SonicTheHedgehog-Genesis','GreenHillZone.Act',('1','3')),
            ('SonicTheHedgehog-Genesis','LabyrinthZone.Act',('1','2','3')),
            ('SonicTheHedgehog-Genesis','MarbleZone.Act',('1','2','3')),
            ('SonicTheHedgehog-Genesis','ScrapBrainZone.Act',('2')),
            ('SonicTheHedgehog-Genesis','SpringYardZone.Act',('2','3')),
            ('SonicTheHedgehog-Genesis','StarLightZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','AquaticRuinZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','CasinoNightZone.Act',('1')),
            ('SonicTheHedgehog2-Genesis','ChemicalPlantZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','EmeraldHillZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','HillTopZone.Act',('1')),
            ('SonicTheHedgehog2-Genesis','MetropolisZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','MysticCaveZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','OilOceanZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','AngelIslandZone.Act',('2')),
            ('SonicAndKnuckles3-Genesis','CarnivalNightZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','DeathEggZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','FlyingBatteryZone.Act',('1')),
            ('SonicAndKnuckles3-Genesis','HydrocityZone.Act',('2')),
            ('SonicAndKnuckles3-Genesis','IcecapZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','LaunchBaseZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','LavaReefZone.Act',('2')),
            ('SonicAndKnuckles3-Genesis','MarbleGardenZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','MushroomHillZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','SandopolisZone.Act',('1','2'))]

    observation_hashes = {}
    info = {}
    frame_stuff = {}
    action = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    f = {}
    for j in envs:
        for k in range(len(j[2])):
            print("Current Game: {}. Current Stage: {}. Current Act: {}.".format(j[0], j[1], k+1,))
            #env = make(game='SonicAndKnuckles3-Genesis', state='SandopolisZone.Act2')
            env = make(game=j[0], state=j[1] + j[2][k])

            model = bebna()
            model.setup(env, observation_hashes)
            obs = env.reset()
            for i in range(1000):
                frame_stuff = {}
                frame_stuff['info'] = info
                frame_stuff['action'] = action.tolist()
                action = model.process_data(obs.astype(np.float32) / 255.0, frame_stuff, f)
                #obs, rew, done, info = env.step(env.action_space.sample())
                obs, rew, done, info = env.step(action)
                #print('Info: {}'.format(info))
                #model.animate_data(obs, obs)
                #env.render()
                if done:
                    obs = env.reset()
            env.close()

    with open("frame_data.json","w") as temp_file:
        json.dump(f, temp_file)
    '''
    env = make(game='SonicAndKnuckles3-Genesis', state='SandopolisZone.Act2')

    model = bebna()
    model.setup(env)
    obs = env.reset()
    while True:
        action = model.process_data(obs, info)
        #obs, rew, done, info = env.step(env.action_space.sample())
        obs, rew, done, info = env.step(action)
        #print('Info: {}'.format(info))
        #model.animate_data(obs, obs)
        env.render()
        if done:
            obs = env.reset()
    '''

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
