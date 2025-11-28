def train_model(args, loader, semantics, unseen_semantics, transformer,  optimizer, logger, epoch_str, epoch):
    # Xác định device từ transformer (hoặc transformer.module nếu dùng DataParallel)
    device = next(transformer.parameters()).device 

    batch_time, Xlosses, CElosses, BCElosses, ATTlosses, KLlosses, MSELosses, accs, token_accs, end = AverageMeter(), \
                                                        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), \
                                                AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), time.time()
    transformer.train()

    loader.dataset.set_return_label_mode('new')
    loader.dataset.set_return_img_mode('original')

    semantics = semantics.to(device) # <-- Đưa semantics lên device chính (GPU 0)
    mse = torch.nn.MSELoss()
    bce= torch.nn.BCELoss()
    kld = torch.nn.KLDivLoss()

    for batch_idx, (img_feat, targets, idx) in enumerate(loader):

        batch = targets.shape[0]  
        # Đưa input data lên device chính (GPU 0)
        img_feat_device = img_feat.to(device) 
        targets_device = targets.to(device)

        source_att_fm, pruned_att_fm, patch_labels, patch_pred = transformer(img_feat_device, epoch=epoch) 

        # ... (Các câu lệnh loss)
        cos_source = torch.einsum('bd,nd->bn', source_att_fm, semantics)
        cos_target = torch.einsum('bd,nd->bn', pruned_att_fm, semantics)

        ce_loss = args.ce_source * F.cross_entropy(cos_source * args.scale, targets_device) + \
                  args.ce_target * F.cross_entropy(cos_target * args.scale, targets_device)

        # ... (Các câu lệnh loss khác)

        # ...
        # Đoạn này không cần thay đổi:
        predict_labels = torch.argmax(cos_target, dim=1)
        predict_tokens = patch_pred > 0.5
        with torch.no_grad():
            accuracy = (predict_labels.cpu() == targets).float().mean().item()
            accs.update(accuracy * 100, batch)
            accuracy = (predict_tokens == patch_labels).float().mean().item()
            token_accs.update(accuracy * 100, batch)

        # ... (Đoạn log)