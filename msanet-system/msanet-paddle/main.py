import argparse
from tools.utils.model_build import ModelBuilder
from tools.utils.phase_build import PhaseBuilder


def parse_args():
    parser = argparse.ArgumentParser(description='Based CenterNet rotation object detection Implementation')
    parser.add_argument('--phase', type=str, help='Phase choice= {train, test, eval}')
    parser.add_argument('--model', type=str, help='String of model name')
    parser.add_argument('--dataset', type=str, help='String of dataset name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # TODO: debug
    phase = ['train', 'eval', 'infer']
    dataset = ['HRSC2016DS', 'KaggleLandShip']
    model = ['MSANet', 'MSANetRBB', 'VectorNet']

    args.phase = phase[0]
    args.dataset = dataset[1]
    args.model = model[1]

    model_builder = ModelBuilder(args).get_model()

    phase_builder = PhaseBuilder(model_builder['config'], model_builder).execute()
