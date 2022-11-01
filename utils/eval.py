import torch
import numpy as np
import torch.nn as nn

def eval_base(net, loader, device, sqc=True):
    net.eval()
    # ce_loss, dice_loss = criterion[0], criterion[1]
    JAp, JAb, JAm, DI, AC, SE, SP, i = [], [], [], [], [], [], [], 0
    total_loss = 0
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    path_set = set()  # 4, 5, c, h, w

    with torch.no_grad():
        for batch in loader:

            Fs = batch['img']  # b, c, t, h, w
            Ms = batch['mask']  # b, c, t, h, w

            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)

            pred_seg = net(Fs)

            output = torch.sigmoid(pred_seg[0])
            true_masks = Ms
            x, y = output.cpu().detach().numpy(), true_masks.cpu().detach().numpy()

            n = x.shape[0]
            for i in range(n):
                results = frame_evaluate(x[i], y[i])
                JAp.append(results[0])
                JAb.append(results[1])
                JAm.append(results[2])
                AC.append(results[3])
                DI.append(results[4])
                SE.append(results[5])
                SP.append(results[6])

    result = {
        "JAp": np.mean(JAp),
        "JAb": np.mean(JAb),
        "JAm": np.mean(JAm),
        "AC": np.mean(AC),
        "DI": np.mean(DI),
        "SE": np.mean(SE),
        "SP": np.mean(SP)
    }
    net.train()
    return total_loss / n_val, result

def eval_pns(net, loader, device, sqc=False):
    net.eval()
    # ce_loss, dice_loss = criterion[0], criterion[1]
    JAp, JAb, JAm, DI, AC, SE, SP, i = [], [], [], [], [], [], [], 0
    total_loss = 0
    n_val = len(loader)  # the number of batch

    with torch.no_grad():
        for batch in loader:
            Fs = batch['img']
            Ms = batch['mask']

            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)

            if len(Ms.shape) > 4:
                b, t, _, h, w = Ms.shape
                Ms = Ms.reshape((b * t, 1, h, w))

            if len(Fs.shape) > 4 and not sqc:
                b, t, _, h, w = Fs.shape
                Fs = Fs.reshape((b * t, 3, h, w))
            pred_seg = net(Fs)
            output = torch.sigmoid(pred_seg)
            true_masks = Ms
            x, y = output.cpu().detach().numpy(), true_masks.cpu().detach().numpy()

            n = x.shape[0]
            for i in range(n):
                results = frame_evaluate(x[i], y[i])
                JAp.append(results[0])
                JAb.append(results[1])
                JAm.append(results[2])
                AC.append(results[3])
                DI.append(results[4])
                SE.append(results[5])
                SP.append(results[6])

    result = {
        "JAp": np.mean(JAp),
        "JAb": np.mean(JAb),
        "JAm": np.mean(JAm),
        "AC": np.mean(AC),
        "DI": np.mean(DI),
        "SE": np.mean(SE),
        "SP": np.mean(SP)
    }
    net.train()
    return total_loss / n_val, result

def eval_pranet(net, loader, device, sqc=False):
    net.eval()
    # ce_loss, dice_loss = criterion[0], criterion[1]
    JAp, JAb, JAm, DI, AC, SE, SP, i = [], [], [], [], [], [], [], 0
    total_loss = 0
    n_val = len(loader)  # the number of batch

    with torch.no_grad():
        for batch in loader:
            Fs = batch['img']
            Ms = batch['mask']

            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)

            pred_seg = net(Fs)
            output = torch.sigmoid(pred_seg['pred'])
            true_masks = Ms
            x, y = output.cpu().detach().numpy(), true_masks.cpu().detach().numpy()

            n = x.shape[0]
            for i in range(n):
                results = frame_evaluate(x[i], y[i])
                JAp.append(results[0])
                JAb.append(results[1])
                JAm.append(results[2])
                AC.append(results[3])
                DI.append(results[4])
                SE.append(results[5])
                SP.append(results[6])

    result = {
        "JAp": np.mean(JAp),
        "JAb": np.mean(JAb),
        "JAm": np.mean(JAm),
        "AC": np.mean(AC),
        "DI": np.mean(DI),
        "SE": np.mean(SE),
        "SP": np.mean(SP)
    }
    net.train()
    return total_loss / n_val, result

def eval_net1(net, loader, device, sqc=True):
    net.eval()
    # ce_loss, dice_loss = criterion[0], criterion[1]
    JAp, JAb, JAm, DI, AC, SE, SP, i = [], [], [], [], [], [], [], 0
    total_loss = 0
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    path_set = set()  # 4, 5, c, h, w

    with torch.no_grad():
        for batch in loader:

            Fs = batch['img']  # b, c, t, h, w
            Ms = batch['mask']  # b, c, t, h, w

            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)

            # Es = torch.zeros_like(Ms)
            # Es[:, :, 0] = Ms[:, :, 0]

            for t in range(0, Fs.shape[2]):

                pred_seg = net(Fs[:, :, t])
                output = torch.sigmoid(pred_seg[0])
                true_masks = Ms[:, :, t, ::]

                x, y = output.cpu().detach().numpy(), true_masks.cpu().detach().numpy()

                n = x.shape[0]
                for i in range(n):
                    results = frame_evaluate(x[i], y[i])
                    JAp.append(results[0])
                    JAb.append(results[1])
                    JAm.append(results[2])
                    AC.append(results[3])
                    DI.append(results[4])
                    SE.append(results[5])
                    SP.append(results[6])

    result = {
        "JAp": np.mean(JAp),
        "JAb": np.mean(JAb),
        "JAm": np.mean(JAm),
        "AC": np.mean(AC),
        "DI": np.mean(DI),
        "SE": np.mean(SE),
        "SP": np.mean(SP)
    }
    net.train()
    return total_loss / n_val, result


def eval_seg(net, loader, device, sqc=True):
    net.eval()
    # ce_loss, dice_loss = criterion[0], criterion[1]
    JAp, JAb, JAm, DI, AC, SE, SP, i = [], [], [], [], [], [], [], 0
    total_loss = 0
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    path_set = set()  # 4, 5, c, h, w

    with torch.no_grad():
        for batch in loader:

            Fs = batch['img']  # b, c, t, h, w
            Ms = batch['mask']  # b, c, t, h, w

            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)

            Es = torch.zeros_like(Ms)
            Es[:, 0] = Ms[:, 0]

            for t in range(0, Fs.shape[1]):

                pred_seg = net(Fs[:, t])

                output = torch.sigmoid(pred_seg[0])
                true_masks = Ms[:, t, ::]
                x, y = output.cpu().detach().numpy(), true_masks.cpu().detach().numpy()

                n = x.shape[0]
                for i in range(n):
                    results = frame_evaluate(x[i], y[i])
                    JAp.append(results[0])
                    JAb.append(results[1])
                    JAm.append(results[2])
                    AC.append(results[3])
                    DI.append(results[4])
                    SE.append(results[5])
                    SP.append(results[6])

    result = {
        "JAp": np.mean(JAp),
        "JAb": np.mean(JAb),
        "JAm": np.mean(JAm),
        "AC": np.mean(AC),
        "DI": np.mean(DI),
        "SE": np.mean(SE),
        "SP": np.mean(SP)
    }
    net.train()
    return total_loss / n_val, result

def eval_seg_sqc(net, loader, device, sqc=True):
    net.eval()
    # ce_loss, dice_loss = criterion[0], criterion[1]
    JAp, JAb, JAm, DI, AC, SE, SP, i = [], [], [], [], [], [], [], 0
    total_loss = 0
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch

    with torch.no_grad():
        for batch in loader:

            Fs = batch['img']  # b, t, c, h, w
            Ms = batch['mask']  # b, t, c, h, w

            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)
            b, c, T, h, w = Fs.shape
            Fs_reshape = Fs.reshape(b * T, c, h, w)
            Ms_reshape = Ms.reshape(b * T, 1, h, w)

            pred_seg = net.my_eval(Fs)

            output = torch.sigmoid(pred_seg)
            output = output.view(b * T, 1, h, w)
            true_masks = Ms_reshape
            x, y = output.cpu().detach().numpy(), true_masks.cpu().detach().numpy()

            n = x.shape[0]
            for i in range(n):
                results = frame_evaluate(x[i], y[i])
                JAp.append(results[0])
                JAb.append(results[1])
                JAm.append(results[2])
                AC.append(results[3])
                DI.append(results[4])
                SE.append(results[5])
                SP.append(results[6])

    result = {
        "JAp": np.mean(JAp),
        "JAb": np.mean(JAb),
        "JAm": np.mean(JAm),
        "AC": np.mean(AC),
        "DI": np.mean(DI),
        "SE": np.mean(SE),
        "SP": np.mean(SP)
    }
    net.train()
    return total_loss / n_val, result

def eval_semi(net, loader, device, sqc=True):
    net.eval()
    JAp, JAb, JAm, DI, AC, SE, SP, i = [], [], [], [], [], [], [], 0
    total_loss = 0
    n_val = len(loader)

    with torch.no_grad():
        for batch in loader:

            Fs = batch['img']  # b, t, c, h, w
            Ms = batch['mask']  # b, t, c, h, w

            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)
            b, c, T, h, w = Fs.shape
            Fs_reshape = Fs.reshape(b * T, c, h, w)
            Ms_reshape = Ms.reshape(b * T, 1, h, w)

            pred_seg = net(Fs)

            output = torch.sigmoid(pred_seg)
            output = output.view(b * T, 1, h, w)
            true_masks = Ms_reshape
            x, y = output.cpu().detach().numpy(), true_masks.cpu().detach().numpy()

            n = x.shape[0]
            for i in range(n):
                results = frame_evaluate(x[i], y[i])
                JAp.append(results[0])
                JAb.append(results[1])
                JAm.append(results[2])
                AC.append(results[3])
                DI.append(results[4])
                SE.append(results[5])
                SP.append(results[6])

    result = {
        "JAp": np.mean(JAp),
        "JAb": np.mean(JAb),
        "JAm": np.mean(JAm),
        "AC": np.mean(AC),
        "DI": np.mean(DI),
        "SE": np.mean(SE),
        "SP": np.mean(SP)
    }
    net.train()
    return total_loss / n_val, result

def eval_prop(net, loader, device, sqc=True):
    net.eval()
    # ce_loss, dice_loss = criterion[0], criterion[1]
    JAp, JAb, JAm, DI, AC, SE, SP, i = [], [], [], [], [], [], [], 0
    total_loss = 0
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    path_set = set()  # 4, 5, c, h, w

    with torch.no_grad():
        for batch in loader:

            Fs = batch['img']  # b, c, t, h, w
            Ms = batch['mask']  # b, c, t, h, w

            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)

            Es = torch.zeros_like(Ms)
            Es[:, :, 0] = Ms[:, :, 0]

            for t in range(1, Fs.shape[2]):
                '''
                # memorize
                Args:
                    frame: torch.Size([b, 3, h, w]
                    masks: torch.Size([b, 1, h, w])
                    num_objects: 1

                Returns:key_M,value_M
                torch.Size([2, 128, 1, 16, 16]) torch.Size([2, 512, 1, 16, 16])
                '''
                prev_key, prev_value = net(Fs[:, :, t - 1], Es[:, :, t - 1])

                if t - 1 == 0:
                    this_keys, this_values = prev_key, prev_value  # only prev memory
                else:
                    this_keys = torch.cat([keys, prev_key], dim=2)
                    this_values = torch.cat([values, prev_value], dim=2)

                '''
                # segment
                Args:
                    frame: [b, c, h, w]
                    keys: [b, 128, t_pre, h/16, w/16]
                    values: [b, 512, t_pre, h/16, w/16]
                    num_objects:

                Returns:
                    [b, 1, h, w]
                '''
                pred_prop = net(Fs[:, :, t], this_keys, this_values)
                # update
                keys, values = this_keys, this_values

                output = torch.sigmoid(pred_prop[0])
                true_masks = Ms[:, :, t, ::]
                x, y = output.cpu().detach().numpy(), true_masks.cpu().detach().numpy()

                n = x.shape[0]
                for i in range(n):
                    results = frame_evaluate(x[i], y[i])
                    JAp.append(results[0])
                    JAb.append(results[1])
                    JAm.append(results[2])
                    AC.append(results[3])
                    DI.append(results[4])
                    SE.append(results[5])
                    SP.append(results[6])

    result = {
        "JAp": np.mean(JAp),
        "JAb": np.mean(JAb),
        "JAm": np.mean(JAm),
        "AC": np.mean(AC),
        "DI": np.mean(DI),
        "SE": np.mean(SE),
        "SP": np.mean(SP)
    }
    net.train()
    return total_loss / n_val, result


def frame_evaluate(x_tmp, target_tmp):

    x_tmp[x_tmp >= 0.5] = 1
    x_tmp[x_tmp <= 0.5] = 0
    x_tmp = np.array(x_tmp, dtype='uint8')

    lesion = x_tmp

    #  calculate TP,TN,FP,FN
    TP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 1)))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 0)))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 0)))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 1)))

    #  calculate JA, Dice, SE-recall, SP
    JAp = TP / (TP + FN + FP + 1e-7)
    JAb = TN / (TN + FN + FP + 1e-7)
    JAm = (JAp + JAb) / 2
    AC = (TP + TN) / (TP + FP + TN + FN + 1e-7)
    DI = 2 * TP / (2 * TP + FN + FP + 1e-7)
    SE = TP / (TP + FN + 1e-7)
    SP = TN / (TN + FP + 1e-7)

    # return sum(JAp_sum), sum(JAb_sum), sum(JAm_sum), sum(AC_sum), sum(DI_sum), sum(SE_sum), sum(SP_sum)
    return JAp, JAb, JAm, AC, DI, SE, SP


def post_process_evaluate(x, target):
    JAp_sum, JAb_sum, JAm_sum, AC_sum, DI_sum, SE_sum, SP_sum = [], [], [], [], [], [], []
    n = x.shape[0]

    for i in range(n):
        x_tmp = x[i]
        target_tmp = target[i]


        x_tmp[x_tmp >= 0.5] = 1
        x_tmp[x_tmp <= 0.5] = 0
        x_tmp = np.array(x_tmp, dtype='uint8')

        lesion = x_tmp

        #  calculate TP,TN,FP,FN
        TP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 1)))
        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 0)))

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 0)))

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 1)))

        #  calculate JA, Dice, SE-recall, SP
        JAp = TP / (TP + FN + FP + 1e-7)
        JAb = TN / (TN + FN + FP + 1e-7)
        JAm = (JAp + JAb) / 2
        AC = (TP + TN) / (TP + FP + TN + FN + 1e-7)
        DI = 2 * TP / (2 * TP + FN + FP + 1e-7)
        SE = TP / (TP + FN + 1e-7)
        SP = TN / (TN + FP + 1e-7)

        JAp_sum.append(JAp)
        JAb_sum.append(JAb)
        JAm_sum.append(JAm)
        AC_sum.append(AC)
        DI_sum.append(DI)
        SE_sum.append(SE)
        SP_sum.append(SP)

    # return sum(JAp_sum), sum(JAb_sum), sum(JAm_sum), sum(AC_sum), sum(DI_sum), sum(SE_sum), sum(SP_sum)
    return JAp_sum, JAb_sum, JAm_sum, AC_sum, DI_sum, SE_sum, SP_sum