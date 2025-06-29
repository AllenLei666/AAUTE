import torch.nn as nn
import torch.nn.functional as F
import torch


class AudioNet(nn.Module):
    def __init__(self, dropout_rate=0.3, kerel_num=16, feature_dim=512):
        super(AudioNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.feature_dim = feature_dim
        self.kernel_num  = kerel_num

        #Audio Def
        # 4 reprents four mic array, the audio spectrogram size is (64,64)
        self.convt1 = nn.Conv2d(4, self.kernel_num, (3, 64))    # time domain
        self.convt2 = nn.Conv2d(4, self.kernel_num, (5, 64))
        self.convf1 = nn.Conv2d(4, self.kernel_num, (64, 3))    # freq domain
        self.convf2 = nn.Conv2d(4, self.kernel_num ,(64, 5))
        self.fc_audio = nn.Linear(3904, self.feature_dim)
        self.dropout1 = nn.Dropout2d(self.dropout_rate)
        self.dropout2 = nn.Dropout2d(self.dropout_rate)
        self.dropout3 = nn.Dropout2d(self.dropout_rate)
        self.dropout4 = nn.Dropout2d(self.dropout_rate)
        self.bn1 = nn.BatchNorm2d(self.kernel_num)
        self.bn2 = nn.BatchNorm2d(self.kernel_num)
        self.bn3 = nn.BatchNorm2d(self.kernel_num)
        self.bn4 = nn.BatchNorm2d(self.kernel_num)
         
        self.position_est1 = nn.Linear(self.feature_dim, 256)
        self.position_est2 = nn.Linear(256, 128)
        self.position_est3 = nn.Linear(128, 3)                           
        self.dropout_p2 = nn.Dropout(0.4)
    def forward(self, x):
        [b, c, row, col] = x.size()

        t1 = F.relu(self.convt1(x))
   
        t1 = self.dropout1(t1).view(b, -1)

        t2 = F.relu(self.convt2(x))
 
        t2 = self.dropout2(t2).view(b, -1)

        f1 = F.relu(self.convf1(x))

        f1 = self.dropout3(f1).view(b, -1)

        f2 = F.relu(self.convf2(x))

        f2 = self.dropout4(f2).view(b, -1)

        feature_tf = torch.cat([t1, t2, f1, f2], dim=-1)
        feature_tf = feature_tf.view(b, -1)
        feature_tf_audio = F.relu(self.fc_audio(feature_tf))   
  
        
        
        # Predict Trajectoy
        position = F.relu(self.position_est1(feature_tf_audio))   
        position  = F.relu(self.position_est2(position)) 
        position = self.dropout_p2(position)
        position  = self.position_est3(position)  
        return position
