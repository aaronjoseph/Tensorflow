import tensorflow as tf 
import traceback 
import contextlib 
import timeit

# Assists with errors 
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught excepted exception \n {}'.format(error_class))
        traceback.print_exc(limit=2)
    except Exception as e:
        raise e
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(error_class))

@tf.function
def add(a,b):
    return(a+b)
add(tf.ones([2,2]), tf.ones([2,2]))

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v,1.0)

tape.gradient(result,v)

@tf.function 
def dense_layer(x,w,b):
    return add(tf.matmul(x,w),b)
dense_layer(tf.ones([3,2]), tf.ones([2,2]), tf.ones([2]))

import timeit
conv_layer = tf.keras.layers.Conv2D(100,3)

@tf.function
def conv_fn(image):
    return conv_layer(image)

image = tf.zeros([1,200,200,100])
conv_layer(image);
conv_fn(image)

# Timing Difference between @tf.function and eager code
print(timeit.timeit(lambda: conv_layer(image), number=100))
print(timeit.timeit(lambda: conv_fn(image), number=100))