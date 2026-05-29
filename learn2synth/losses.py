from torch import nn
import torch
import inspect
from . import utils


def _dot(x, y):
    """Dot product along the last dimension"""
    return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def _make_activation(activation):
    if isinstance(activation, str):
        activation = getattr(nn, activation)
    if inspect.isclass(activation):
        # Softmax needs dim=1 for (B, C, *spatial) tensors
        if activation is nn.Softmax:
            activation = activation(dim=1)
        else:
            activation = activation()
    elif callable(activation):
        pass
    else:
        activation = None
    return activation


class Loss(nn.Module):
    """Base class for losses"""

    def __init__(self, reduction='mean'):
        """
        Parameters
        ----------
        reduction : {'mean', 'sum'} or callable
            Reduction to apply across batch elements
        """
        super().__init__()
        self.reduction = reduction

    def reduce(self, x):
        if not self.reduction:
            return x
        if isinstance(self.reduction, str):
            if self.reduction.lower() == 'mean':
                return x.mean()
            if self.reduction.lower() == 'sum':
                return x.sum()
            raise ValueError(f'Unknown reduction "{self.reduction}"')
        if callable(self.reduction):
            return self.reduction(x)
        raise ValueError(f'Don\'t know what to do with reduction: '
                         f'{self.reduction}')


class DiceLoss(Loss):
    r"""Soft Dice Loss

    By default, each class is weighted identically.
    The `weighted` mode allows classes to be weighted by frequency.

    References
    ----------
    ..  "V-Net: Fully convolutional neural networks for volumetric
         medical image segmentation"
        Milletari, Navab and Ahmadi
        3DV (2016)
        https://arxiv.org/abs/1606.04797
    ..  "Generalised dice overlap as a deep learning loss function for
         highly unbalanced segmentations"
        Sudre, Li, Vercauteren, Ourselin and Cardoso
        DLMIA (2017)
        https://arxiv.org/abs/1707.03237
    ..  "The Dice loss in the context of missing or empty labels:
         introducing $\Phi$ and $\epsilon$"
        Tilborghs, Bertels, Robben, Vandermeulen and Maes
        MICCAI (2022)
        https://arxiv.org/abs/2207.09521
    """

    def __init__(self, square=True, weighted=False, labels=None,
                 eps=None, reduction='mean', activation=None):
        """

        Parameters
        ----------
        square : bool, default=True
            Square the denominator in SoftDice.
        weighted : bool or list[float], default=False
            If True, weight the Dice of each class by its frequency in the
            reference. If a list, use these weights for each class.
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        eps : float or list[float], default=1/K
            Stabilization of the Dice loss.
            Optimally, should be equal to each class' expected frequency
            across the whole dataset. See Tilborghs et al.
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        activation : nn.Module or str
            Activation to apply to the prediction before computing the loss
        """
        super().__init__(reduction)
        self.square = square
        self.weighted = weighted
        self.labels = labels
        self.eps = eps
        self.activation = _make_activation(activation)

    def forward_onehot(self, pred, ref, mask, weights, eps):

        nb_classes = pred.shape[1]
        if ref.shape[1] != nb_classes:
            raise ValueError(f'Number of classes not consistent. '
                             f'Expected {nb_classes} but got {ref.shape[1]}.')

        ref = ref.to(pred)
        if mask is not None:
            pred = pred * mask
            ref = ref * mask
        pred = pred.reshape([*pred.shape[:2], -1])  # [B, C, N]
        ref = ref.reshape([*ref.shape[:2], -1])  # [B, C, N]

        # Compute SoftDice
        inter = _dot(pred, ref)  # [B, C]
        if self.square:
            pred = pred.square()
            ref = ref.square()
        pred = pred.sum(-1)  # [B, C]
        ref = ref.sum(-1)  # [B, C]
        union = pred + ref
        loss = (2 * inter + eps) / (union + eps)

        # Simple or weighted average
        if weights is not False:
            if weights is True:
                weights = ref / ref.sum(dim=1, keepdim=True)
            loss = loss * weights
            loss = loss.sum(-1)
        else:
            loss = loss.mean(-1)

        # Minibatch reduction
        loss = 1 - loss
        return self.reduce(loss)

    def forward_labels(self, pred, ref, mask, weights, eps):

        nb_classes = pred.shape[1]
        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            pred1 = pred[:, index]
            eps1 = eps[index]
            ref1 = (ref == label).squeeze(1)
            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask

            pred1 = pred1.reshape([len(pred1), -1])  # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])  # [B, N]

            # Compute SoftDice
            inter = (pred1 * ref1).sum(-1)  # [B]
            if self.square:
                pred1 = pred1.square()
            pred1 = pred1.sum(-1)  # [B]
            ref1 = ref1.sum(-1)  # [B]
            union = pred1 + ref1
            loss1 = (2 * inter + eps1) / (union + eps1)

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1
                else:
                    weight1 = float(weights[index])
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += 1
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        loss = 1 - loss
        return self.reduce(loss)

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)
        nvox = pred.shape[2:].numel()

        eps = self.eps or 1 / nb_classes
        eps = utils.make_vector(eps, nb_classes, **backend)
        eps = eps * nvox

        # prepare weights
        weighted = self.weighted
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = utils.make_vector(weighted, nb_classes, **backend)

        if ref.dtype.is_floating_point:
            return self.forward_onehot(pred, ref, mask, weighted, eps)
        else:
            return self.forward_labels(pred, ref, mask, weighted, eps)


class CatLoss(Loss):
    r"""Weighted categorical cross-entropy."""

    def __init__(self, weighted=False, labels=None, reduction='mean', activation=None):
        super().__init__(reduction)
        self.weighted = weighted
        self.labels = labels
        self.reduction = reduction
        self.activation = _make_activation(activation)

    def forward_onehot(self, pred, ref, mask, weights):
        nb_classes = pred.shape[1]
        if ref.shape[1] != nb_classes:
            raise ValueError(f'Number of classes not consistent.')

        ref = ref.to(pred)

        if mask is not None:
            pred = pred * mask
            ref = ref * mask

        pred = pred.reshape([*pred.shape[:2], -1])  # [B, C, N]
        ref = ref.reshape([*ref.shape[:2], -1])  # [B, C, N]

        # Compute dot(ref, log(pred))
        loss = _dot(pred, ref)  # [B, C]
        ref_sum = ref.sum(-1).clamp_min(1e-5)  # Prevent division by zero
        loss = loss / ref_sum  # [B, C]

        if weights is not False:
            if weights is True:
                weights = ref_sum / ref_sum.sum(dim=1, keepdim=True).clamp_min(1e-5)
            loss = loss * weights
            loss = loss.sum(-1)
        else:
            loss = loss.mean(-1)

        # Standard CE Negation
        return self.reduce(loss.neg_())

    def forward_labels(self, pred, ref, mask, weights):
        nb_classes = pred.shape[1]
        labels = self.labels or list(range(nb_classes))

        loss = 0
        sum_weights = 0

        for index, label in enumerate(labels):
            if label is None:
                continue

            pred1 = pred[:, index]
            ref1 = (ref == label).squeeze(1).float()

            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask

            pred1 = pred1.reshape([len(pred1), -1])  # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])  # [B, N]

            loss1 = (pred1 * ref1).sum(-1)  # [B]
            ref_sum = ref1.sum(-1).clamp_min(1e-5)  # out-of-place, consistent with forward_onehot
            loss1 = loss1 / ref_sum

            if weights is not False:
                if weights is True:
                    weight1 = ref_sum
                else:
                    weight1 = float(weights[index])

                loss1 = loss1 * weight1
                sum_weights += weight1
            else:
                sum_weights += 1

            loss += loss1

        loss = loss / sum_weights
        return self.reduce(loss.neg_())

    def forward(self, pred, ref, mask=None):
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)

        # FIX: Clamp probabilities BEFORE log to avoid log(0) -> -inf
        pred = pred.clamp(min=1e-7, max=1.0)
        pred = pred.log()

        weighted = self.weighted

        if not torch.is_tensor(weighted) and not weighted:
            weighted = False

        if not isinstance(weighted, bool):
            weighted = utils.make_vector(weighted, nb_classes, **backend)

        if ref.dtype.is_floating_point:
            return self.forward_onehot(pred, ref, mask, weighted)
        else:
            return self.forward_labels(pred, ref, mask, weighted)


class CatMSELoss(Loss):
    """Mean Squared Error between one-hots."""

    def __init__(self, weighted=False, labels=None, reduction='mean',
                 activation=None):
        """

        Parameters
        ----------
        weighted : bool or list[float], default=False
            If True, weight the Dice of each class by its size in the
            reference. If a list, use these weights for each class.
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        activation : nn.Module or str
            Activation to apply to the prediction before computing the loss
        """
        super().__init__(reduction)
        self.weighted = weighted
        self.labels = labels
        self.reduction = reduction
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.activation = activation

    def forward_onehot(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        if ref.shape[1] != nb_classes:
            raise ValueError(f'Number of classes not consistent. '
                             f'Expected {nb_classes} but got {ref.shape[1]}.')

        ref = ref.to(pred)
        if mask is not None:
            pred = pred * mask
            ref = ref * mask
            mask = mask.reshape([*mask.shape[:2], -1])

        pred = pred.reshape([*pred.shape[:2], -1])  # [B, C, N]
        ref = ref.reshape([*ref.shape[:2], -1])  # [B, C, N]
        loss = pred - ref
        loss = _dot(loss, loss)  # [B, C]
        loss = loss / (mask.sum(-1) if mask is not None else pred.shape[-1])

        # Simple or weighted average
        if weights is not False:
            if weights is True:
                weights = ref / ref.sum(dim=1, keepdim=True)
            loss = loss * weights
            loss = loss.sum(-1)
        else:
            loss = loss.mean(-1)

        # Minibatch reduction
        return self.reduce(loss)

    def forward_labels(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            pred1 = pred[:, index]
            ref1 = (ref == label).squeeze(1)
            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask
                mask1 = mask.reshape([len(mask), -1])

            pred1 = pred1.reshape([len(pred1), -1])  # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])  # [B, N]

            # Compute SoftDice
            loss1 = pred1 - ref1
            loss1 = _dot(loss1, loss1)
            loss1 = loss1 / (mask1.sum(-1) if mask is not None
                             else pred1.shape[-1])

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1
                else:
                    weight1 = float(weights[index])
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += 1
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        return self.reduce(loss)

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)

        # prepare weights
        weighted = self.weighted
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = utils.make_vector(weighted, nb_classes, **backend)

        if ref.dtype.is_floating_point:
            return self.forward_onehot(pred, ref, mask, weighted)
        else:
            return self.forward_labels(pred, ref, mask, weighted)


class LogitMSELoss(Loss):
    """Mean Squared Error between logits and target positive/negative values."""

    def __init__(self, target=5, weighted=False, labels=None, reduction='mean',
                 activation=None):
        """

        Parameters
        ----------
        target : float
            Target value when the ground truth is True.
        weighted : bool or list[float] or 'inv', default=False
            If True, weight the score of each class by its frequency in
            the reference.
            If 'inv', weight the score of each class by its inverse
            frequency in the reference.
            If a list, use these weights for each class.
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        activation : nn.Module or str
            Activation to apply to the prediction before computing the loss
        """
        super().__init__(reduction)
        self.weighted = weighted
        self.labels = labels
        self.reduction = reduction
        self.target = target
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.activation = activation

    def forward_onehot(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        if ref.shape[1] != nb_classes:
            raise ValueError(f'Number of classes not consistent. '
                             f'Expected {nb_classes} but got {ref.shape[1]}.')

        ref = ref.to(pred)
        if mask is not None:
            pred = pred * mask
            ref = ref * mask
            mask = mask.reshape([*mask.shape[:2], -1])

        pred = pred.reshape([*pred.shape[:2], -1])  # [B, C, N]
        ref = ref.reshape([*ref.shape[:2], -1])  # [B, C, N]
        loss = pred + (1 - 2 * ref) * self.target
        loss = _dot(loss, loss)  # [B, C]
        loss = loss / (mask.sum(-1) if mask is not None else pred.shape[-1])

        # Simple or weighted average
        if weights is not False:
            if weights is True:
                weights = ref.sum(dim=-1)
                weights = weights / weights.sum(dim=-1, keepdim=True)
            elif isinstance(weights, str) and weights[0].lower() == 'i':
                weights = ref.sum(dim=-1)
                weights = ref.shape[-1] - weights
                weights = weights / weights.sum(dim=-1, keepdim=True)
            loss = (loss * weights).sum(-1)
        else:
            loss = loss.mean(-1)

        # Minibatch reduction
        return self.reduce(loss)

    def forward_labels(self, pred, ref, mask, weights):

        nb_classes = pred.shape[1]
        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            pred1 = pred[:, index]
            ref1 = (ref == label).squeeze(1)
            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask
                mask1 = mask.reshape([len(mask), -1])

            pred1 = pred1.reshape([len(pred1), -1])  # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])  # [B, N]

            # Compute SoftDice
            loss1 = pred1 + (1 - 2 * ref1) * self.target
            loss1 = _dot(loss1, loss1)
            loss1 = loss1 / (mask1.sum(-1) if mask is not None
                             else pred1.shape[-1])

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1.sum(-1)
                elif isinstance(weights, str) and weights[0].lower() == 'i':
                    weight1 = ref1.shape[-1] - ref1.sum(-1)
                else:
                    weight1 = float(weights[index])
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += 1
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        return self.reduce(loss)

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)

        # prepare weights
        weighted = self.weighted
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = utils.make_vector(weighted, nb_classes, **backend)

        if ref.dtype.is_floating_point:
            return self.forward_onehot(pred, ref, mask, weighted)
        else:
            return self.forward_labels(pred, ref, mask, weighted)


class DiceCELoss(Loss):
    r"""Compound loss: Weighted sum of Soft Dice Loss and Categorical Cross-Entropy.

    Computes: lambda_dice * DiceLoss + lambda_ce * CrossEntropyLoss.
    The `activation` is applied once before passing the result to the
    sub-losses.

    Parameters
    ----------
    lambda_dice : float, default=1.0
        Scaling factor for the Dice Loss term.
    lambda_ce : float, default=1.0
        Scaling factor for the Cross-Entropy Loss term.
    square : bool, default=True
        Square the denominator in SoftDice. Passed to DiceLoss.
    weighted : bool or list[float], default=False
        If True, weight the Dice/CE of each class by its frequency in the
        reference. If a list, use these weights for each class.
    labels : list[int], default=range(nb_class)
        Label corresponding to each one-hot class. Only used if the
        reference is an integer label map.
    eps : float or list[float], default=1/K
        Stabilization of the Dice loss. Passed to DiceLoss.
    reduction : {'mean', 'sum', None} or callable, default='mean'
        Type of reduction to apply across minibatch elements.
    activation : nn.Module or str
        Activation to apply to the prediction before computing the loss
        (e.g., 'Softmax').
    """

    def __init__(self, lambda_dice=1.0, lambda_ce=1.0, square=True, weighted=False, labels=None, eps=None, reduction='mean', activation=None):
        super().__init__(reduction)
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.activation = _make_activation(activation)

        # Initialize sub-losses.
        # We pass activation=None because the activation is handled
        # in the forward pass of this wrapper class to ensure consistency.
        self.dice   = DiceLoss(square=square, weighted=weighted, labels=labels, eps=eps, reduction=reduction, activation=None)
        self.ce     = CatLoss(                weighted=weighted, labels=labels,          reduction=reduction, activation=None)

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes (logits).
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The weighted sum of Dice and Cross-Entropy losses.
        """
        # Apply activation once (e.g., Softmax) to get probabilities
        if self.activation:
            pred = self.activation(pred)

        # Compute sub-losses using the activated predictions
        l_dice = self.dice(pred, ref, mask)
        l_ce = self.ce(pred, ref, mask)

        # Compute weighted sum
        return self.lambda_dice * l_dice + self.lambda_ce * l_ce


class FocalTverskyLoss(Loss):
    """
    Focal Tversky Loss for highly imbalanced data (e.g. small lesions).

    Reference:
    Abraham et al. "A Novel Focal Tversky Loss Function with Improved
    Attention U-Net for Lesion Segmentation" (ISBI 2019)
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=4.0 / 3.0, weighted=False,
                 labels=None, eps=1e-6, reduction='mean', activation=None):
        """
        Parameters
        ----------
        alpha : float
            Weight for False Positives (Penalize FP).
            Higher alpha -> Higher Precision.
        beta : float
            Weight for False Negatives (Penalize FN).
            Higher beta -> Higher Recall (Crucial for FCD).
        gamma : float
            Focal parameter.
            gamma=1.0 -> Standard Tversky Loss.
            gamma>1.0 -> Focal Tversky (Focus on hard examples).
        """
        super().__init__(reduction)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weighted = weighted
        self.labels = labels
        self.eps = eps
        self.activation = _make_activation(activation)

    def forward(self, pred, ref, mask=None):
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]

        # 1. Prepare Reference (One-Hot)
        if not ref.dtype.is_floating_point:
            ref_onehot = torch.zeros_like(pred)
            ref_onehot.scatter_(1, ref.long(), 1)
            ref = ref_onehot

        if mask is not None:
            pred = pred * mask
            ref = ref * mask

        # 2. Flatten [Batch, Class, Voxels]
        pred = pred.reshape(pred.shape[0], nb_classes, -1)
        ref = ref.reshape(ref.shape[0], nb_classes, -1)

        # 3. Compute True Positives (TP), False Positives (FP), False Negatives (FN)
        TP = (pred * ref).sum(-1)
        FP = (pred * (1 - ref)).sum(-1)
        FN = ((1 - pred) * ref).sum(-1)

        # 4. Tversky Index
        # TI = TP / (TP + alpha*FP + beta*FN)
        tversky_index = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)

        # 5. Focal Tversky Loss
        # Loss = (1 - TI)^gamma
        loss = (1 - tversky_index).pow(self.gamma)

        # 6. Weighting (Optional)
        if self.weighted:
            if self.weighted is True:
                # Inverse volume weighting
                w = ref.sum(-1)
                w = 1.0 / (w * w + 1e-6)
                w = w / w.sum(-1, keepdim=True)
            else:
                w = torch.tensor(self.weighted, device=pred.device)
            loss = loss * w
            loss = loss.sum(-1)
        else:
            loss = loss.mean(-1)

        return self.reduce(loss)
