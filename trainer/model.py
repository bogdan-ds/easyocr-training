import torch.nn as nn
from modules.feature_extraction import VGG_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_size = hidden_size
        self.num_class = num_class


        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(self.input_channel, self.output_channel)
        self.FeatureExtraction_output = self.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size, self.hidden_size),
            BidirectionalLSTM(self.hidden_size, self.hidden_size, self.hidden_size))
        self.SequenceModeling_output = self.hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, self.num_class)

    def forward(self, input, text, is_train=True):

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
