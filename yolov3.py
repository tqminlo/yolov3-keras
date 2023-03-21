import tensorflow as tf
from keras.layers import *
from keras.models import Model


class YOLOv3:
    def __init__(self, size=416, num_classes=80):
        self.size = size
        self.num_classes = num_classes


    def conv_bn_leky(self, filters, kernel_size, strides, padding, x):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        return x


    def res_unit(self, filter1, filter2, x):
        x1 = self.conv_bn_leky(filters=filter1, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x)
        x1 = self.conv_bn_leky(filters=filter2, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x1)
        x = Add()([x, x1])
        return x


    def res_n(self, n, filter0, filter1, filter2, x):
        x = self.conv_bn_leky(filters=filter0, kernel_size=(3, 3), strides=(2, 2), padding='same', x=x)
        for i in range(n):
            x = self.res_unit(filter1, filter2, x)
        return x


    def darknet_53(self, x):
        x = self.conv_bn_leky(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x)
        x = self.res_n(n=1, filter0=64, filter1=32, filter2=64, x=x)
        x = self.res_n(n=2, filter0=128, filter1=64, filter2=128, x=x)
        x = self.res_n(n=8, filter0=256, filter1=128, filter2=256, x=x)
        x2 = x
        x = self.res_n(n=8, filter0=512, filter1=256, filter2=512, x=x)
        x1 = x
        x = self.res_n(n=4, filter0=1024, filter1=512, filter2=1024, x=x)
        return x, x1, x2


    def structure(self):
        inputs = Input(shape=(self.size, self.size, 3))
        x0, x1, x2 = self.darknet_53(inputs)

        x0 = self.conv_bn_leky(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x0)
        x0 = self.conv_bn_leky(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x0)
        x0 = self.conv_bn_leky(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x0)
        x0 = self.conv_bn_leky(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x0)
        x0 = self.conv_bn_leky(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x0)
        x01 = x0
        x0 = self.conv_bn_leky(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x0)
        y0 = Conv2D(filters=255, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(x0)

        x01 = self.conv_bn_leky(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x01)
        x01 = UpSampling2D(size=(2, 2), interpolation='nearest')(x01)
        x1 = Concatenate(axis=-1)([x1, x01])
        x1 = self.conv_bn_leky(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x1)
        x1 = self.conv_bn_leky(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x1)
        x1 = self.conv_bn_leky(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x1)
        x1 = self.conv_bn_leky(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x1)
        x1 = self.conv_bn_leky(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x1)
        x12 = x1
        x1 = self.conv_bn_leky(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x1)
        y1 = Conv2D(filters=255, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(x1)

        x12 = self.conv_bn_leky(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x12)
        x12 = UpSampling2D(size=(2, 2), interpolation='nearest')(x12)
        x2 = Concatenate(axis=-1)([x2, x12])
        x2 = self.conv_bn_leky(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x2)
        x2 = self.conv_bn_leky(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x2)
        x2 = self.conv_bn_leky(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x2)
        x2 = self.conv_bn_leky(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x2)
        x2 = self.conv_bn_leky(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', x=x2)
        x2 = self.conv_bn_leky(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', x=x2)
        y2 = Conv2D(filters=255, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(x2)

        model = Model(inputs=inputs, outputs=[y0, y1, y2])
        return model

    def model(self):
        '''
        Demo output with B=1 (not B=3 in original), so each cell in each map returns a tensor with shape (5+C,)
        '''
        model = self.structure()
        y0 = model.outputs[0]
        y1 = model.outputs[1]
        y2 = model.outputs[2]
        y0 = Conv2D(filters=5+self.num_classes, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(y0)
        y1 = Conv2D(filters=5+self.num_classes, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(y1)
        y2 = Conv2D(filters=5+self.num_classes, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(y2)
        model = Model(inputs=model.inputs, outputs=[y0, y1, y2])
        return model


if __name__ == "__main__":
    model = YOLOv3(size=416, num_classes=3).model()
    model.summary()