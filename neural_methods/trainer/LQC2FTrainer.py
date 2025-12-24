import logging
import os
import numpy as np
import torch
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from evaluation.metrics import calculate_metrics, calculate_metrics_return
from evaluation.metrics import calculate_hr
from neural_methods.model.LQC2F import Label_Quantizer, C2F_model
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from evaluation.post_process import _detrend, _bandpass_filter, get_hr
from neural_methods.loss.TorchLossComputer import TorchLossComputer, Hybrid_Loss_Batched
from tqdm import tqdm
from neural_methods.loss.TorchLossComputer import Hybrid_Loss


class LQC2FTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.epoch_for_valid = 0
        self.data_dict = {}

        self.device = torch.device(config.DEVICE)
        self.max_epoch_num_lq = config.TRAIN.LQ_EPOCHS
        self.max_epoch_num_c2f = config.TRAIN.C2F_EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.diff_flag = 0
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1

        # HR logs accumulated across training (for later MAE computation)
        self.hr_logs = {"q": [], "label": [], "bpf": []}

        self.LQ_min_valid_loss = [None, None, None, None, None]
        self.C2F_min_valid_loss = None
        
        self.LQ_best_epoch = [0, 0, 0, 0, 0]
        self.C2F_best_epoch = 0

        self.LQ_pretrained = config.TRAIN.LQ_PRETRAINED
        self.C2F_pretrained = config.TRAIN.C2F_PRETRAINED

        self.neg_loss = Neg_Pearson()

        if self.LQ_pretrained:
            self.LQ_pretrained_path = config.TRAIN.LQ_PRETRAINED_PATH

        if self.C2F_pretrained:
            self.C2F_pretrained_path = config.TRAIN.C2F_PRETRAINED_PATH
        
        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])

            # Reconstruction criterion (batched) for soft-reconstructed signals vs targets
            self.recon_criterion = Hybrid_Loss_Batched()

            self.LQ_model_1 = Label_Quantizer(num_codebooks=2).to(self.device)
            self.LQ_model_2 = Label_Quantizer(num_codebooks=4).to(self.device)
            self.LQ_model_3 = Label_Quantizer(num_codebooks=8).to(self.device)
            self.LQ_model_4 = Label_Quantizer(num_codebooks=16).to(self.device)
            self.LQ_model_5 = Label_Quantizer(num_codebooks=32).to(self.device)

            self.C2F_model = C2F_model().to(self.device)

            self.LQ_model_1 = torch.nn.DataParallel(self.LQ_model_1, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.LQ_model_2 = torch.nn.DataParallel(self.LQ_model_2, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.LQ_model_3 = torch.nn.DataParallel(self.LQ_model_3, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.LQ_model_4 = torch.nn.DataParallel(self.LQ_model_4, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.LQ_model_5 = torch.nn.DataParallel(self.LQ_model_5, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.C2F_model = torch.nn.DataParallel(self.C2F_model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))          
            
            self.LQ_optimizer_1 = optim.AdamW(self.LQ_model_1.parameters(), lr=float(config.TRAIN.LR_LQ[0]), weight_decay=0)
            self.LQ_optimizer_2 = optim.AdamW(self.LQ_model_2.parameters(), lr=float(config.TRAIN.LR_LQ[1]), weight_decay=0)
            self.LQ_optimizer_3 = optim.AdamW(self.LQ_model_3.parameters(), lr=float(config.TRAIN.LR_LQ[2]), weight_decay=0)
            self.LQ_optimizer_4 = optim.AdamW(self.LQ_model_4.parameters(), lr=float(config.TRAIN.LR_LQ[3]), weight_decay=0)
            self.LQ_optimizer_5 = optim.AdamW(self.LQ_model_5.parameters(), lr=float(config.TRAIN.LR_LQ[4]), weight_decay=0)

            self.C2F_optimizer = optim.AdamW(
                list(self.C2F_model.parameters()),
                lr=config.TRAIN.LR_C2F,
                weight_decay=1e-5
            )

            self.LQ_scheduler_1 = torch.optim.lr_scheduler.OneCycleLR(self.LQ_optimizer_1, max_lr=float(config.TRAIN.LR_LQ[0]), epochs=self.max_epoch_num_lq, steps_per_epoch=self.num_train_batches)
            self.LQ_scheduler_2 = torch.optim.lr_scheduler.OneCycleLR(self.LQ_optimizer_2, max_lr=float(config.TRAIN.LR_LQ[1]), epochs=self.max_epoch_num_lq, steps_per_epoch=self.num_train_batches)
            self.LQ_scheduler_3 = torch.optim.lr_scheduler.OneCycleLR(self.LQ_optimizer_3, max_lr=float(config.TRAIN.LR_LQ[2]), epochs=self.max_epoch_num_lq, steps_per_epoch=self.num_train_batches)
            self.LQ_scheduler_4 = torch.optim.lr_scheduler.OneCycleLR(self.LQ_optimizer_4, max_lr=float(config.TRAIN.LR_LQ[3]), epochs=self.max_epoch_num_lq, steps_per_epoch=self.num_train_batches)
            self.LQ_scheduler_5 = torch.optim.lr_scheduler.OneCycleLR(self.LQ_optimizer_5, max_lr=float(config.TRAIN.LR_LQ[4]), epochs=self.max_epoch_num_lq, steps_per_epoch=self.num_train_batches)
            
            self.C2F_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.C2F_optimizer, max_lr=config.TRAIN.LR_C2F, epochs=self.max_epoch_num_c2f, steps_per_epoch=self.num_train_batches)
            
            self.list_LQ_model = [self.LQ_model_1, self.LQ_model_2, self.LQ_model_3, self.LQ_model_4, self.LQ_model_5]
            self.list_LQ_scheduler = [self.LQ_scheduler_1, self.LQ_scheduler_2, self.LQ_scheduler_3, self.LQ_scheduler_4, self.LQ_scheduler_5]
            self.list_LQ_optimizer = [self.LQ_optimizer_1, self.LQ_optimizer_2, self.LQ_optimizer_3, self.LQ_optimizer_4, self.LQ_optimizer_5]
            
            
            
        elif config.TOOLBOX_MODE == "only_test": 
            self.LQ_model_1 = Label_Quantizer(num_codebooks=2).to(self.device)
            self.LQ_model_2 = Label_Quantizer(num_codebooks=4).to(self.device)
            self.LQ_model_3 = Label_Quantizer(num_codebooks=8).to(self.device)
            self.LQ_model_4 = Label_Quantizer(num_codebooks=16).to(self.device)
            self.LQ_model_5 = Label_Quantizer(num_codebooks=32).to(self.device)

            self.C2F_model = C2F_model().to(self.device)
            
            self.LQ_model_1 = torch.nn.DataParallel(self.LQ_model_1, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.LQ_model_2 = torch.nn.DataParallel(self.LQ_model_2, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.LQ_model_3 = torch.nn.DataParallel(self.LQ_model_3, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.LQ_model_4 = torch.nn.DataParallel(self.LQ_model_4, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.LQ_model_5 = torch.nn.DataParallel(self.LQ_model_5, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.C2F_model = torch.nn.DataParallel(self.C2F_model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            
            self.list_LQ_model = [self.LQ_model_1, self.LQ_model_2, self.LQ_model_3, self.LQ_model_4, self.LQ_model_5]
            
        else:
            raise ValueError("DeepPhys trainer initialized in incorrect toolbox mode!")


    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        config = self.config
        
        print('')
        print("===Testing===")
        C2F_rPPG_predictions = dict() 
        labels = dict() 
        
        if self.config.TOOLBOX_MODE == "only_test":
            if self.LQ_pretrained and self.C2F_pretrained:
                print(f"Testing uses pretrained LQ model from {self.LQ_pretrained_path}")
                for num_LQ in range(1, 6):
                    print('')
                    print("Load Pre-trained {}th LQ model".format(num_LQ))
                    self.list_LQ_model[num_LQ-1].load_state_dict(torch.load(self.config.TRAIN.LQ_PRETRAINED_PATH[num_LQ-1]))

                print(f"Testing uses pretrained C2F model from {self.C2F_pretrained_path}")
                checkpoint = torch.load(self.C2F_pretrained_path)
                self.C2F_model.load_state_dict(checkpoint['C2F_model'])
                # Inject pretrained codebooks into backbone after loading pretrained LQ models
                with torch.no_grad():
                    codebook_1 = self.LQ_model_1.module.quantizer.embeddings.weight
                    codebook_2 = self.LQ_model_2.module.quantizer.embeddings.weight
                    codebook_3 = self.LQ_model_3.module.quantizer.embeddings.weight
                    codebook_4 = self.LQ_model_4.module.quantizer.embeddings.weight
                    self.C2F_model.module.load_codebooks(codebook_1, codebook_2, codebook_3, codebook_4)
                
            else:
                raise ValueError("Testing uses pretrained LQ and C2F model!")

        else: # train_and_test
            if self.config.TEST.USE_LAST_EPOCH:
                for num_LQ in range(1, 6):
                    print("Load pretrained {}th LQ model".format(num_LQ))
                    LQ_last_model_path = os.path.join(self.model_dir, self.model_file_name + '_' + str(num_LQ) + 'th_LQ' + '_Epoch' + str(self.max_epoch_num_lq - 1) + '.pth')
                    print(LQ_last_model_path)
                    print("Load the last {}th LQ model (Last epoch: {})".format(num_LQ, self.max_epoch_num_lq - 1))
                    self.list_LQ_model[num_LQ-1].load_state_dict(torch.load(LQ_last_model_path))

                # Inject pretrained codebooks into backbone after loading last LQ models
                with torch.no_grad():
                    codebook_1 = self.LQ_model_1.module.quantizer.embeddings.weight
                    codebook_2 = self.LQ_model_2.module.quantizer.embeddings.weight
                    codebook_3 = self.LQ_model_3.module.quantizer.embeddings.weight
                    codebook_4 = self.LQ_model_4.module.quantizer.embeddings.weight
                    self.C2F_model.module.load_codebooks(codebook_1, codebook_2, codebook_3, codebook_4)


                C2F_last_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_rPPG' + '_Epoch' + str(self.max_epoch_num_c2f - 1) + '.pth')
                print("Testing uses last C2F model epoch: {}".format(self.max_epoch_num_c2f - 1))
                print(C2F_last_model_path)
                checkpoint = torch.load(C2F_last_model_path)
                self.C2F_model.load_state_dict(checkpoint['C2F_model'])
            else:
                if not self.LQ_pretrained: # From training stage 1 and stage 2 
                    for num_LQ in range(1, 6):
                        print("Best trained {}th LQ epoch: {}".format(num_LQ, self.LQ_best_epoch[num_LQ-1]))
                        LQ_best_model_path = os.path.join(self.model_dir, self.model_file_name + '_' + str(num_LQ) + 'th_LQ' + '_Epoch' + str(self.LQ_best_epoch[num_LQ-1]) + '.pth')
                        print(LQ_best_model_path)
                        print("Load the best {}th LQ model (Best epoch: {})".format(num_LQ, self.LQ_best_epoch[num_LQ-1]))
                        self.list_LQ_model[num_LQ-1].load_state_dict(torch.load(LQ_best_model_path))
                else: # From training stage 2
                    print(f"Testing uses pretrained LQ model from {self.LQ_pretrained_path}")
                    for num_LQ in range(1, 6):
                        print("Load Pre-trained {}th LQ model".format(num_LQ))
                        self.list_LQ_model[num_LQ-1].load_state_dict(torch.load(self.config.TRAIN.LQ_PRETRAINED_PATH[num_LQ-1]))

                    # Inject pretrained codebooks into backbone after loading pretrained LQ models
                    with torch.no_grad():
                        codebook_1 = self.LQ_model_1.module.quantizer.embeddings.weight
                        codebook_2 = self.LQ_model_2.module.quantizer.embeddings.weight
                        codebook_3 = self.LQ_model_3.module.quantizer.embeddings.weight
                        codebook_4 = self.LQ_model_4.module.quantizer.embeddings.weight
                        self.C2F_model.module.load_codebooks(codebook_1, codebook_2, codebook_3, codebook_4)


                C2F_best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_rPPG' + '_Epoch' + str(self.C2F_best_epoch) + '.pth')
                print("Testing uses best C2F model epoch: {}".format(self.C2F_best_epoch))
                print(C2F_best_model_path)
                checkpoint = torch.load(C2F_best_model_path)
                self.C2F_model.load_state_dict(checkpoint['C2F_model'])



        self.LQ_model_1 = self.LQ_model_1.to(self.config.DEVICE)
        self.LQ_model_2 = self.LQ_model_2.to(self.config.DEVICE)
        self.LQ_model_3 = self.LQ_model_3.to(self.config.DEVICE)
        self.LQ_model_4 = self.LQ_model_4.to(self.config.DEVICE)
        self.C2F_model = self.C2F_model.to(self.config.DEVICE)

        self.LQ_model_1.eval()
        self.LQ_model_2.eval()
        self.LQ_model_3.eval()
        self.LQ_model_4.eval()
        self.C2F_model.eval()

        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            tbar = tqdm(data_loader["test"])
            for _, test_batch in enumerate(tbar):
                tbar.set_description("Testing")
                batch_size = test_batch[0].shape[0]
                data_test = test_batch[0].float()
                labels_original_test = test_batch[1].float()


                labels_original_test = (labels_original_test - labels_original_test.mean(dim=-1, keepdim=True)) / (labels_original_test.std(dim=-1, keepdim=True) + 1e-6)
                
                data_test = data_test.to(self.device)
                labels_original_test = labels_original_test.to(self.device)
                

                # C2F Prediction   
                final_token_head, _, _, _, _ = self.C2F_model(data_test)
                

                # Soft reconstruction from final token head (use codebook_5)
                c5 = self.LQ_model_5.module.quantizer.embeddings.weight.detach().to(self.device)
                logits_5 = -torch.cdist(final_token_head.reshape(-1, 1), c5)
                probs_5 = F.softmax(logits_5, dim=-1)
                soft_rec_flat_5 = torch.matmul(probs_5, c5)  # (B*T, 1)
                Bt, Tt = final_token_head.size()
                soft_rec_5 = soft_rec_flat_5.view(Bt, Tt)
                # normalize for downstream visualization/metrics
                soft_rec_5 = (soft_rec_5 - torch.mean(soft_rec_5, dim=-1, keepdim=True)) / (torch.std(soft_rec_5, dim=-1, keepdim=True) + 1e-6)


                labels_original_test = labels_original_test.cpu()
                soft_rec_5 = soft_rec_5.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in C2F_rPPG_predictions.keys():
                        C2F_rPPG_predictions[subj_index] = dict() 
                        labels[subj_index] = dict()

                    C2F_rPPG_predictions[subj_index][sort_index] = soft_rec_5[idx]
                    labels[subj_index][sort_index] = labels_original_test[idx]

        print('')
        print('C2F rPPG prediction results')
        calculate_metrics(C2F_rPPG_predictions, labels, self.config) 
        if self.config.TEST.OUTPUT_SAVE_DIR: 
            self.save_test_outputs(C2F_rPPG_predictions, labels, self.config) 

    def save_LQ_model(self, index, num_LQ):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_' + str(num_LQ) + 'th_LQ' + '_Epoch' + str(index) + '.pth')
        torch.save(self.list_LQ_model[num_LQ-1].state_dict(), model_path)
        print(f"{num_LQ}-th LQ model saved to {model_path}")

    def save_C2F_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_rPPG' + '_Epoch' + str(index) + '.pth')
        torch.save({'C2F_model': self.C2F_model.state_dict()}, model_path)
        print(f"C2F model saved to {model_path}")


    def save_codebook_usage_histogram(self, codebook_usage, epoch, num_LQ, save_dir="codebook_usage"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(codebook_usage)), codebook_usage, color='blue', alpha=0.7)
        plt.title(f"{num_LQ}-th LQ Codebook Usage at Epoch {epoch}")
        plt.xlabel("Codebook Index")
        plt.ylabel("Usage Count")
        plt.grid(True)

        save_path = os.path.join(save_dir, f"{num_LQ}-th_codebook_usage_epoch_{epoch}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def visualize_codebook(self, codebook, epoch, num_LQ, save_dir="codebook_visualizations"):
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)

        codebook_np = codebook.cpu().detach().numpy() 

        if codebook_np.shape[1] == 1:
            plt.figure(figsize=(8, 8))  # 정사각형 형태로 설정
            plt.scatter(codebook_np[:, 0], np.zeros_like(codebook_np[:, 0]), c='blue', alpha=0.6)
            for i, x in enumerate(codebook_np[:, 0]):
                plt.text(x, 0, str(i), fontsize=10, ha='right', va='bottom')
            plt.title(f"{num_LQ}-th LQ 1D Codebook Scatter Plot at Epoch {epoch}")
            plt.xlabel("Codebook Value")
            plt.ylabel("Fixed Y (0)")
            plt.grid(True)
            save_path = os.path.join(save_dir, f"{num_LQ}-th_codebook_epoch_{epoch}_1d_scatter.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
    
    def update_codebook_usage(self, encoding_index):
        for u in np.unique(encoding_index):
            c = np.sum(encoding_index == u)
            self.codebook_usage_buffer[u] += c


    def data_augmentation(self,data,labels,index1,index2):
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))
        rand1_vals = np.random.random(N)
        rand2_vals = np.random.random(N)
        for idx in range(N):
            index = index1[idx] + index2[idx]
            rand1 = rand1_vals[idx]
            rand2 = rand2_vals[idx]
            if rand1 < 0.5 :
                if index in self.data_dict:
                    gt_hr_fft = self.data_dict[index]
                else:
                    gt_hr_fft, _  = calculate_hr(labels[idx], labels[idx] , diff_flag = self.diff_flag , fs=self.config.VALID.DATA.FS)
                    self.data_dict[index] = gt_hr_fft
                    
                if gt_hr_fft > 90: 
                    rand3 = random.randint(0, D//2-1)
                    even_indices = torch.arange(0, D, 2)
                    odd_indices = even_indices + 1
                    data_aug[:, even_indices, :, :, :] = data[:, rand3 + even_indices// 2, :, :, :]
                    labels_aug[:, even_indices] = labels[:, rand3 + even_indices // 2]
                    data_aug[:, odd_indices, :, :, :] = (data[:, rand3 + odd_indices // 2, :, :, :] + data[:, rand3 + (odd_indices // 2) + 1, :, :, :]) / 2
                    labels_aug[:, odd_indices] = (labels[:, rand3 + odd_indices // 2] + labels[:, rand3 + (odd_indices // 2) + 1]) / 2
                elif gt_hr_fft < 75 :
                    data_aug[:, :D//2, :, :, :] = data[:, ::2, :, :, :]
                    labels_aug[:, :D//2] = labels[:, ::2]
                    data_aug[:, D//2:, :, :, :] = data_aug[:, :D//2, :, :, :]
                    labels_aug[:, D//2:] = labels_aug[:, :D//2]
                else :
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]                                      
            else :
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]
        data_aug = torch.tensor(data_aug).float()
        labels_aug = torch.tensor(labels_aug).float()
        if rand2 < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        return data_aug, labels_aug