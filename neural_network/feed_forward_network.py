from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from funcy import first, last
import numpy as np
import pandas as pd
import timeit
import math

BIN_SIZE = 15

def get_bins(race_data):
    """ Group races and create bins (time ranges) of BIN_SIZE. For each
        bin find out pct of racers in that bin and avg time of that bin.
        Also assign bin number to identify racers and their bin they fall
        into later on.
    """
    bin_data = []
    race_groups = race_data.groupby('race_id')

    for race_id, race_group in race_groups:

        top_75_percentile = race_group[
            race_group.final_time < race_group.final_time.quantile(.75)]

        # Skip races with missing data.
        if len(top_75_percentile) == 0:
            continue

        bins = pd.cut(top_75_percentile.final_time, BIN_SIZE, right=False)

        # fastest = time.strftime(
        #     '%H:%M:%S', time.gmtime(min(top_75_percentile.final_time)))
        # slowest = time.strftime(
        #     '%H:%M:%S', time.gmtime(max(top_75_percentile.final_time)))

        # print "fastest =>", fastest
        # print "slowest =>", slowest

        bin_number = 0

        for bin_key, bin_group in top_75_percentile.groupby(bins):

            bin_number += 1

            population_pct = len(bin_group) / float(len(top_75_percentile))
            bin_avg_time = bin_group.final_time.mean()

            if math.isnan(bin_avg_time):
                # Yes Ugly. Pandas bin key is a string.
                # This split gives us bin's lower/upper range time.
                lower_range = float(first(bin_key.split(',')).strip('['))
                upper_range = float(last(bin_key.split(',')).strip(')'))

                bin_avg_time = np.mean([lower_range, upper_range])

            bin_data.append({'race_id': int(race_id),
                             'bin_number': bin_number,
                             'population_pct': population_pct,
                             'bin_avg_time': bin_avg_time
                             })
    return bin_data

def _sum_square_error(actual, desired):
    error = 0.
    for i in range(len(desired)):
        for j in range(len(desired[i])):
            error = error + \
                ((actual[i])[j] - (desired[i])[j]) * (
                    (actual[i])[j] - (desired[i])[j])

    return error


def get_supervised_dataset(race_data, race_factors):

    race_bins = get_bins(race_data)
    race_bin_groups = pd.DataFrame.from_dict(race_bins).groupby('race_id')

    # Input, ouput
    data_set = SupervisedDataSet(6, 15)

    for race_id, race_bin in race_bin_groups:

        # Skipe bins with fewer than 10% race population
        if not np.count_nonzero(race_bin.population_pct) > 10:
            continue

        race_factor = race_factors[race_factors.race_id == race_id]

        # If race has missing factor data then skip
        if race_factor.empty:
            continue

        input_factors = [first(race_factor.high_temp) / 100.0,
                         first(race_factor.low_temp) / 100.0,
                         first(race_factor.high_humidity) / 100.0,
                         first(race_factor.low_humidity) / 100.0,
                         first(race_factor.starting_elevation) / 10000.0,
                         first(race_factor.gross_elevation_gain) / 10000.0
                         ]

        output_factors = race_bin.population_pct.tolist()

        data_set.appendLinked(input_factors, output_factors)

    return data_set


def create_feedforward_network(supervised_dataset):
    network = FeedForwardNetwork()

    inLayer = LinearLayer(6)
    hiddenLayer = SigmoidLayer(5)
    outLayer = LinearLayer(15)

    network.addInputModule(inLayer)
    network.addModule(hiddenLayer)
    network.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    network.addConnection(in_to_hidden)
    network.addConnection(hidden_to_out)

    # Activate network. This is very important.
    network.sortModules()

    return network


def train_network():

    start = timeit.default_timer()

    # Read data
    race_data = pd.read_csv('../data/half_ironman_data_v1.csv')
    race_factors = pd.read_csv('../data/half_ironman_race_factors.csv')

    # Prepare input data
    supervised_dataset = get_supervised_dataset(race_data, race_factors)

    # Create network
    network = create_feedforward_network(supervised_dataset)

    train_data, test_data = supervised_dataset.splitWithProportion(0.9)

    trainer = BackpropTrainer(network, dataset=train_data)

    # Train our network
    trainer.trainEpochs(1)

    # check network accuracy
    print _sum_square_error(network.activateOnDataset(dataset=train_data), train_data['target'])
    print _sum_square_error(network.activateOnDataset(dataset=test_data), test_data['target'])

    print 'Execution time =>', timeit.default_timer() - start, 'secs'


if __name__ == "__main__":
    train_network()
