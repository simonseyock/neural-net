from neural_net import *

def draw_results(computed, samples, targets, title, path, file):
    pyplot.axes(xlim=(-10, 10))
    pyplot.plot(samples, computed, '.')
    pyplot.plot(samples, targets, '.')
    pyplot.suptitle('{}\naverage quadratic error: {}'.format(title, average_quadratic_error(computed, targets_vec)))
    if path is not None:
        pyplot.savefig(path + file, bbox_inches='tight')
    else:
        pyplot.show()


def draw_error(avg_error, title, path, file):
    # show development of quadratic error
    generations = len(avg_error)
    gen_space = np.arange(0, generations, 1)
    pyplot.axes(xlim=(0, generations - 1), yscale='log', ylim=(0.01, 1))
    pyplot.plot(gen_space, avg_error)
    pyplot.suptitle('{}\ndevelopment of average quadratic error\nlog scale'.format(title))
    if path is not None:
        pyplot.savefig(path + file, bbox_inches='tight')
    else:
        pyplot.show()


def save_weights(net, path, file):
    if path is not None:
        np.save(path + file, net.weights)


output_folder = './out'
generations = 100
learn_rate = .0001
momentum = 0
layer_sizes = [20, 1]

# stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# path = '{}/{}/'.format(output_folder, stamp)
# os.makedirs(path)
path = None

# create samples
samples = np.linspace(-10, 10, 1001, dtype=data_type)
targets = target_function(samples)

samples_vec = np.array([[e] for e in samples], dtype=data_type)
targets_vec = np.array([[e] for e in targets], dtype=data_type)

# create net
net = NeuralNet(1)
last_size = 1
for size in layer_sizes[:-1]:
    net.add_layer(SumLayer(last_size, size, capped_fermi, fermi_back))
    last_size = size
net.add_layer(SumLayer(last_size, layer_sizes[-1], identity, identity_back))

save_weights(net, path, 'weights_1.txt')

# before training
computed = net.calculate(samples_vec)
draw_results(computed, samples, targets, 'Before training', path, 'graph_1.png')

# # save weights
# stored_weights = np.copy(net.weights)
#
# # training only second layer
# avg_error = net.train(samples_vec, targets_vec, generations, learn_rate, 1)
# draw_error(avg_error, 'After training only second layer', path, 'graph_2.png')
# computed = net.calculate(samples_vec)
# draw_results(computed, samples, targets, 'After training only second layer', path, 'graph_3.png')
#
# save_weights(net, path, 'weights_2.txt')
#
# # reset net
# net.weights = stored_weights

if path is not None:
    with open(path + 'meta.txt', 'w') as file:
        file.write('layers: {}\n'.format('-'.join(['1'] + [str(s) for s in layer_sizes])))
        file.write('generations: {}\n'.format(generations))
        file.write('learn_rate: {}\n'.format(learn_rate))
        file.write('momentum: {}\n'.format(momentum))
        file.write('using capped fermi\n')

# training both layers
# avg_error = learn_full_batches(net, samples_vec, targets_vec, generations, learn_rate, momentum=momentum)
avg_error = learn_random_order(net, samples_vec, targets_vec, generations, learn_rate, momentum=momentum)
draw_error(avg_error, 'After training both layers', path, 'graph_4.png')
computed = net.calculate(samples_vec)
draw_results(computed, samples, targets, 'After training both layers', path, 'graph_5.png')

save_weights(net, path, 'weights_3.txt')