from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from NeuralNetwork import NeuralNetwork
from preprocess_data import preprocess_data
import random
import pygame
import argparse
import csv

def main(args):
    # Parse arguments
    file_name = args.input
    category = args.attribute
    hidden_layers = args.hiddennodes
    iterations = args.iterations
    repeat = args.repeat
    output_path = args.output

    running = True
    paused = False
    drawing = args.visualise

    # Set up PyGame
    if drawing:
        (width, height) = (1200, 500)
        screen = pygame.display.set_mode((width, height))
        pygame.font.init()

    a = 0  # Counter
    tests = []
    while running:
        # Test network and reset
        if a % iterations == 0:
            if a != 0:
                # Run test and save results
                test = test_network(nn, testing_data)
                tests.append(test)

            if a < iterations * repeat:
                # Preprocess and divide data into two sets, ratio 2:1
                training_data, testing_data, headings, categories = preprocess_data(file_name, category)
                # Initialise neural network
                keys, values = list(training_data.keys()), list(training_data.values())
                input_layers, output_layers = len(keys[0]), len(values[0])
                nn = NeuralNetwork(input_layers, hidden_layers, output_layers, args.learningrate)
                # Get current weights
                weights = nn.get_weights()
                weights_ih = weights['input-hidden'].data
                weights_oh = weights['hidden-output'].data
            else:
                # Save results
                path = output_path + 'results.csv'
                file = open(path, 'w')
                writer = csv.writer(file)
                writer.writerow(['successes', 'failures', 'success%'])
                t_successes = t_failures = 0
                for test in tests:
                    test = list(test)
                    test.append("{:.2f}".format((test[0] / (test[1] + test[0])) * 100))
                    writer.writerow(test)
                    t_successes += test[0]
                    t_failures += test[1]

                print("Total successes: " + str(t_successes) + "\t\tTotal fails: " + str(
                    t_failures) + "\t\tTotal success rate: " + "{:.2f}".format((t_successes / (t_failures + t_successes)) * 100) + "%")

                running = False

        # Create visualisation
        if drawing:
            for event in pygame.event.get():
                # Stop on close
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    # Pause/unpause when space is pressed
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            # Black background
            screen.fill((0, 0, 0))

            # Calculate node positions
            c = input_layers * 40 + (input_layers - 1) * 10
            offset_i = 20 + (height - c) // 2
            c = hidden_layers * 40 + (hidden_layers - 1) * 10
            offset_h = 20 + (height - c) // 2
            c = output_layers * 40 + (output_layers - 1) * 10
            offset_o = 20 + (height - c) // 2

            # Calculate min and max values for input-hidden weights
            min_weight = min([abs(item) for sublist in weights_ih for item in sublist])
            max_weight = max([abs(item) for sublist in weights_ih for item in sublist])
            # Draw input-hidden weights
            for i in range(input_layers):
                for j in range(hidden_layers):
                    weight = weights_ih[j][i]
                    if weight < 0:
                        weight = (abs(weight) - min_weight) / (max_weight - min_weight)
                        shade = int(255 * weight)
                        pygame.draw.aaline(screen, (0, shade, 0), (300, offset_i + 50 * i), (width / 2, offset_h + 50 * j))
                    else:
                        weight = (abs(weight) - min_weight) / (max_weight - min_weight)
                        shade = int(255 * weight)
                        pygame.draw.aaline(screen, (shade, 0, 0), (300, offset_i + 50 * i), (width / 2, offset_h + 50 * j))

            # Calculate min and max values for hidden-output weights
            min_weight = min([abs(item) for sublist in weights_oh for item in sublist])
            max_weight = max([abs(item) for sublist in weights_oh for item in sublist])
            # Draw hidden-output weights
            for i in range(hidden_layers):
                for j in range(output_layers):
                    weight = weights_oh[j][i]
                    if weight < 0:
                        weight = (abs(weight) - min_weight) / (max_weight - min_weight)
                        shade = int(255 * weight)
                        pygame.draw.aaline(screen, (0, shade, 0), (width / 2, offset_h + 50 * i), (width - 300, offset_o + 50 * j))
                    else:
                        weight = (abs(weight) - min_weight) / (max_weight - min_weight)
                        shade = int(255 * weight)
                        pygame.draw.aaline(screen, (shade, 0, 0), (width / 2, offset_h + 50 * i), (width - 300, offset_o + 50 * j))

            if a % iterations != 0:
                activations = nn.get_activations()

                # Draw input nodes
                activation = [item for sublist in activations['input'].data for item in sublist]
                for i in range(input_layers):
                    shade = int(activation[i] * 255)
                    pygame.draw.circle(screen, (70, 70, 70), (300, offset_i + 50 * i), 20)
                    pygame.draw.circle(screen, (shade, shade, shade), (300, offset_i + 50 * i), 15)
                    # Label nodes
                    myfont = pygame.font.SysFont('Consolas', 20)
                    textsurface = myfont.render(headings[i], False, (255, 255, 255))
                    rect = textsurface.get_rect()
                    rect.right = 270
                    rect.top = offset_i - 10 + 50 * i
                    screen.blit(textsurface, rect)

                # Draw hidden nodes
                activation = [item for sublist in activations['hidden'].data for item in sublist]
                for i in range(hidden_layers):
                    shade = int(activation[i] * 255)
                    pygame.draw.circle(screen, (100, 100, 100), (width // 2, offset_h + 50 * i), 20)
                    pygame.draw.circle(screen, (shade, shade, shade), (width // 2, offset_h + 50 * i), 15)

                # Draw output nodes
                activation = [item for sublist in activations['output'].data for item in sublist]
                for i in range(output_layers):
                    shade = int(activation[i] * 255)
                    pygame.draw.circle(screen, (100, 100, 100), (width - 300, offset_o + 50 * i), 20)
                    pygame.draw.circle(screen, (shade, shade, shade), (width - 300, offset_o + 50 * i), 15)
                    # Label nodes
                    myfont = pygame.font.SysFont('Consolas', 20)
                    textsurface = myfont.render(categories[i], False, (255, 255, 255))
                    screen.blit(textsurface, (width - 270, offset_o - 10 + 50 * i))

            # Show iterations
            myfont = pygame.font.SysFont('Consolas', 20)
            textsurface = myfont.render('Iterations: ' + str(a % iterations) + '\t Repeat: ' + str(a // iterations),
                                        False, (255, 255, 255))
            screen.blit(textsurface, (width - 500, 30))
            pygame.display.flip()

        # Train network
        inputs, target = random.choice(list(training_data.items()))
        if not paused:
            nn.train(list(inputs), target)
            a += 1



def test_network(nn, testing_data):
    """ Test the network on a set of data """
    successes = fails = 0
    for data in testing_data:
        test = nn.feed_forward(list(data))
        for i in range(3):
            if float(test[i]) > 0.7:
                test[i] = 1
            else:
                test[i] = 0

        if test == testing_data[data]:
            successes += 1
        else:
            fails += 1

    print("Successes: " + str(successes) + "\t\tFails: " + str(
        fails) + "\t\tSuccess rate: " + "{:.2f}".format((successes / (fails + successes)) * 100) + "%")

    return successes, fails

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='An implementation of the multi-layer perceptron algorithm.')
    parser.add_argument("-lr","--learningrate", default=5, type=int, help="Set the learning rate for backpropagation.")
    parser.add_argument("-iter","--iterations", default=5000, type=int, help="Set the number of epochs the algorithm trains for.")
    parser.add_argument("-hn","--hiddennodes", default=5, type=int, help="Set the number of nodes in the hidden layer.")
    parser.add_argument("-r","--repeat", default=1, type=int, help="Set the number of times to repeat.")
    parser.add_argument("-o","--output", default="", type=str, help="Set the location of the output file.")
    parser.add_argument("-v","--visualise", action="store_true")

    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input file name', required=True) #, default='beer_data.csv'
    requiredNamed.add_argument('-a', '--attribute', help='Classification column name.', required=True) #, default='style'

    args = parser.parse_args()
    main(args)