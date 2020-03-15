# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# Author: Nicolas Hug

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

from libc.math cimport exp, fabs, log1p

from .common cimport Y_DTYPE_C
from .common cimport G_H_DTYPE_C


def _update_gradients_least_squares(
        G_H_DTYPE_C [::1] gradients,  # OUT
        const Y_DTYPE_C [::1] y_true,  # IN
        const Y_DTYPE_C [::1] raw_predictions):  # IN

    cdef:
        int n_samples
        int i

    n_samples = raw_predictions.shape[0]
    for i in prange(n_samples, schedule='static', nogil=True):
        # Note: a more correct exp is 2 * (raw_predictions - y_true)
        # but since we use 1 for the constant hessian value (and not 2) this
        # is strictly equivalent for the leaves values.
        gradients[i] = raw_predictions[i] - y_true[i]


def _update_gradients_hessians_least_squares(
        G_H_DTYPE_C [::1] gradients,  # OUT
        G_H_DTYPE_C [::1] hessians,  # OUT
        const Y_DTYPE_C [::1] y_true,  # IN
        const Y_DTYPE_C [::1] raw_predictions,  # IN
        const Y_DTYPE_C [::1] sample_weight):  # IN

    cdef:
        int n_samples
        int i

    n_samples = raw_predictions.shape[0]
    for i in prange(n_samples, schedule='static', nogil=True):
        # Note: a more correct exp is 2 * (raw_predictions - y_true) * sample_weight
        # but since we use 1 for the constant hessian value (and not 2) this
        # is strictly equivalent for the leaves values.
        gradients[i] = (raw_predictions[i] - y_true[i]) * sample_weight[i]
        hessians[i] = sample_weight[i]


def _update_gradients_hessians_least_absolute_deviation(
        G_H_DTYPE_C [::1] gradients,  # OUT
        G_H_DTYPE_C [::1] hessians,  # OUT
        const Y_DTYPE_C [::1] y_true,  # IN
        const Y_DTYPE_C [::1] raw_predictions,  # IN
        const Y_DTYPE_C [::1] sample_weight):  # IN

    cdef:
        int n_samples
        int i

    n_samples = raw_predictions.shape[0]
    for i in prange(n_samples, schedule='static', nogil=True):
        # gradient = sign(raw_predicition - y_pred) * sample_weight
        gradients[i] = sample_weight[i] * (2 *
                        (y_true[i] - raw_predictions[i] < 0) - 1)
        hessians[i] = sample_weight[i]


def _update_gradients_least_absolute_deviation(
        G_H_DTYPE_C [::1] gradients,  # OUT
        const Y_DTYPE_C [::1] y_true,  # IN
        const Y_DTYPE_C [::1] raw_predictions):  # IN

    cdef:
        int n_samples
        int i

    n_samples = raw_predictions.shape[0]
    for i in prange(n_samples, schedule='static', nogil=True):
        # gradient = sign(raw_predicition - y_pred)
        gradients[i] = 2 * (y_true[i] - raw_predictions[i] < 0) - 1


def _update_gradients_hessians_binary_crossentropy(
        G_H_DTYPE_C [::1] gradients,  # OUT
        G_H_DTYPE_C [::1] hessians,  # OUT
        const Y_DTYPE_C [::1] y_true,  # IN
        const Y_DTYPE_C [::1] raw_predictions,  # IN
        const Y_DTYPE_C [::1] sample_weight):  # IN
    cdef:
        int n_samples
        Y_DTYPE_C p_i  # proba that ith sample belongs to positive class
        int i

    n_samples = raw_predictions.shape[0]
    if sample_weight is None:
        for i in prange(n_samples, schedule='static', nogil=True):
            p_i = _cexpit(raw_predictions[i])
            gradients[i] = p_i - y_true[i]
            hessians[i] = p_i * (1. - p_i)
    else:
        for i in prange(n_samples, schedule='static', nogil=True):
            p_i = _cexpit(raw_predictions[i])
            gradients[i] = (p_i - y_true[i]) * sample_weight[i]
            hessians[i] = p_i * (1. - p_i) * sample_weight[i]


def _update_gradients_hessians_categorical_crossentropy(
        G_H_DTYPE_C [:, ::1] gradients,  # OUT
        G_H_DTYPE_C [:, ::1] hessians,  # OUT
        const Y_DTYPE_C [::1] y_true,  # IN
        const Y_DTYPE_C [:, ::1] raw_predictions,  # IN
        const Y_DTYPE_C [::1] sample_weight):  # IN
    cdef:
        int prediction_dim = raw_predictions.shape[0]
        int n_samples = raw_predictions.shape[1]
        int k  # class index
        int i  # sample index
        Y_DTYPE_C sw
        # p[i, k] is the probability that class(ith sample) == k.
        # It's the softmax of the raw predictions
        Y_DTYPE_C [:, ::1] p = np.empty(shape=(n_samples, prediction_dim))
        Y_DTYPE_C p_i_k

    if sample_weight is None:
        for i in prange(n_samples, schedule='static', nogil=True):
            # first compute softmaxes of sample i for each class
            for k in range(prediction_dim):
                p[i, k] = raw_predictions[k, i]  # prepare softmax
            _compute_softmax(p, i)
            # then update gradients and hessians
            for k in range(prediction_dim):
                p_i_k = p[i, k]
                gradients[k, i] = p_i_k - (y_true[i] == k)
                hessians[k, i] = p_i_k * (1. - p_i_k)
    else:
        for i in prange(n_samples, schedule='static', nogil=True):
            # first compute softmaxes of sample i for each class
            for k in range(prediction_dim):
                p[i, k] = raw_predictions[k, i]  # prepare softmax
            _compute_softmax(p, i)
            # then update gradients and hessians
            sw = sample_weight[i]
            for k in range(prediction_dim):
                p_i_k = p[i, k]
                gradients[k, i] = (p_i_k - (y_true[i] == k)) * sw
                hessians[k, i] = (p_i_k * (1. - p_i_k)) * sw

def _get_linear_constraint(Y_DTYPE_C [:, ::1] A): # OUT
    cdef:
        int K = A.shape[1]
        np.ndarray first_row = np.append(np.array([1, -1]),
                                         np.zeros(K-3, dtype=np.float64))
        np.ndarray first_col = np.append(np.array([1]),
                                         np.zeros(K-3, dtype=np.float64))
        Y_DTYPE_C [:, ::1] D = toeplitz(first_col, first_row)
    A[0, 0] = 1.
    A[1:, 1:] = D

@cython.boundscheck(False)
def _update_gradients_hessians_all_threshold(
        G_H_DTYPE_C [:, ::1] gradients,  # OUT
        G_H_DTYPE_C [:, ::1] hessians,  # OUT
        G_H_DTYPE_C [:, ::1] mixed_partials,  # OUT
        const Y_DTYPE_C [::1] y_true,  # IN
        const Y_DTYPE_C [::1] raw_predictions,  # IN
        const Y_DTYPE_C [::1] sample_weight,  # IN
        const Y_DTYPE_C [::1] theta):  # IN
    cdef:
        int K = theta.shape[0] + 1
        int n_samples = raw_predictions.shape[0]
        int k  # class index
        int i  # sample index
        Y_DTYPE_C sw
        Y_DTYPE_C s_i_k
        Y_DTYPE_C d_i_k
        Y_DTYPE_C z_i_k

    if sample_weight is None:
        for i in prange(n_samples, schedule='static', nogil=True):
            for k in range(K-1):
                s_i_k = _sgn(y_true[i] - k)
                d_i_k = raw_predictions[i] - theta[k]
                z_i_k = s_i_k * d_i_k
                gradients[k, i] = -s_i_k * _cexpit(-z_i_k)
                hessians[k, i] = _sigmdx(z_i_k)
                mixed_partials[k, i] = _sigsigm(z_i_k)
    else:
        for i in prange(n_samples, schedule='static', nogil=True):
            sw = sample_weight[i]
            for k in range(K-1):
                s_i_k = _sgn(y_true[i] - k)
                d_i_k = raw_predictions[i] - theta[k]
                z_i_k = s_i_k * d_i_k
                gradients[k, i] = -s_i_k * _cexpit(-z_i_k) * sw
                hessians[k, i] = _sigmdx(z_i_k) * sw
                mixed_partials[k, i] = _sigsigm(z_i_k) * sw

def _AT_objective(
        Y_DTYPE_C [::1] loss,  # OUT
        const Y_DTYPE_C [::1] y_true,  # IN
        const Y_DTYPE_C [::1] raw_predictions,  # IN
        const Y_DTYPE_C [::1] theta):  # IN
    cdef:
        int K = theta.shape[0] + 1
        int n_samples = raw_predictions.shape[0]
        int i
        int k
        Y_DTYPE_C s_i_k
        Y_DTYPE_C d_i_k
        Y_DTYPE_C z_i_k
        Y_DTYPE_C sum_i

    for i in prange(n_samples, schedule='static', nogil=True):
        sum_i = 0.
        for k in range(K-1):
            s_i_k = _sgn(y_true[i] - k)
            d_i_k = raw_predictions[i] - theta[k]
            z_i_k = s_i_k * d_i_k
            sum_i = sum_i + _log1pexp(-z_i_k)
        loss[i] = sum_i

def _loss_AT(
        const Y_DTYPE_C [::1] x0,
        const Y_DTYPE_C [::1] y_true,
        const Y_DTYPE_C [::1] sample_weight):
    cdef:
        Y_DTYPE_C y_init = x0[0]
        Y_DTYPE_C [::1] theta = x0[1:].copy()
        int n_samples = y_true.shape[0]
        int K = x0.shape[0]
        int i
        int k
        Y_DTYPE_C sw
        Y_DTYPE_C s_i_k
        Y_DTYPE_C d_i_k
        Y_DTYPE_C z_i_k
        Y_DTYPE_C _sum = 0

    for i in prange(n_samples, schedule='static', nogil=True):
        sw = sample_weight[i]
        for k in range(K-1):
            s_i_k = _sgn(y_true[i] - k)
            d_i_k = y_init - theta[k]
            z_i_k = s_i_k * d_i_k
            _sum += _log1pexp(-z_i_k) * sw
    return _sum

def _jac_AT(
        const Y_DTYPE_C [::1] x0,
        const Y_DTYPE_C [::1] y_true,
        const Y_DTYPE_C [::1] sample_weight):

    cdef:
        Y_DTYPE_C y_init = x0[0]
        Y_DTYPE_C [::1] theta = x0[1:].copy()
        int n_samples = y_true.shape[0]
        int K = x0.shape[0]
        int i
        int k
        Y_DTYPE_C sw
        Y_DTYPE_C s_i_k
        Y_DTYPE_C d_i_k
        Y_DTYPE_C z_i_k
        Y_DTYPE_C p_i_k
        Y_DTYPE_C [::1] jac_out = np.zeros(shape=K)

    for i in prange(n_samples, schedule='static', nogil=True):
        sw = sample_weight[i]
        for k in range(K-1):
            s_i_k = _sgn(y_true[i] - k)
            d_i_k = y_init - theta[k]
            z_i_k = s_i_k * d_i_k
            p_i_k = s_i_k * _cexpit(-z_i_k)
            jac_out[k+1] += p_i_k * sw
            jac_out[0] += -p_i_k * sw
    return jac_out

def _hess_AT(
        const Y_DTYPE_C [::1] x0,
        const Y_DTYPE_C [::1] y_true,
        const Y_DTYPE_C [::1] sample_weight):
    cdef:
        Y_DTYPE_C y_init = x0[0]
        Y_DTYPE_C [::1] theta = x0[1:].copy()
        int n_samples = y_true.shape[0]
        int K = x0.shape[0]
        int i
        int k
        Y_DTYPE_C sw
        Y_DTYPE_C s_i_k
        Y_DTYPE_C d_i_k
        Y_DTYPE_C z_i_k
        Y_DTYPE_C p_i_k
        Y_DTYPE_C q_i_k
        Y_DTYPE_C [:, ::1] hess_out = np.zeros(shape=(K, K))

    for i in prange(n_samples, schedule='static', nogil=True):
        sw = sample_weight[i]
        for k in range(K-1):
            s_i_k = _sgn(y_true[i] - k)
            d_i_k = y_init - theta[k]
            z_i_k = s_i_k * d_i_k
            p_i_k = _sigmdx(z_i_k)
            q_i_k = _sigsigm(z_i_k)
            hess_out[0][0] += p_i_k * sw
            hess_out[k+1][0] += q_i_k * sw
            hess_out[0][k+1] += q_i_k * sw
            hess_out[k+1][k+1] += p_i_k * sw
    return hess_out

cdef inline void _compute_softmax(Y_DTYPE_C [:, ::1] p, const int i) nogil:
    """Compute softmaxes of values in p[i, :]."""
    # i needs to be passed (and stays constant) because otherwise Cython does
    # not generate optimal code

    cdef:
        Y_DTYPE_C max_value = p[i, 0]
        Y_DTYPE_C sum_exps = 0.
        unsigned int k
        unsigned prediction_dim = p.shape[1]

    # Compute max value of array for numerical stability
    for k in range(1, prediction_dim):
        if max_value < p[i, k]:
            max_value = p[i, k]

    for k in range(prediction_dim):
        p[i, k] = exp(p[i, k] - max_value)
        sum_exps += p[i, k]

    for k in range(prediction_dim):
        p[i, k] /= sum_exps

@cython.cdivision(True)
cdef inline Y_DTYPE_C _cexpit(const Y_DTYPE_C x) nogil:
    """Custom stable expit (logistic sigmoid function)"""
    cdef:
        Y_DTYPE_C z

    if x >= 0:
        z = exp(-x)
        return 1. / (1. + z)
    else:
        z = exp(x)
        return z / (1. + z)

cdef inline Y_DTYPE_C _log1pexp(const Y_DTYPE_C x) nogil:
    """log(1 + exp(x))"""

    if x <= -37:
        return exp(x)
    elif -37 < x and x <= 18:
        return log1p(exp(x))
    elif 18 < x and x <= 33.3:
        return x + exp(-x)
    else:
        return x

@cython.cdivision(True)
cdef inline Y_DTYPE_C _sigmdx(const Y_DTYPE_C x) nogil:
    """(2 * sigmoid(x) - 1) / (2 * x)"""
    cdef:
        Y_DTYPE_C z, t

    if fabs(x) < 1e-10:
        return 0.25
    else:
        if x >= 0:
            z = exp(-x)
            t = (1. - z) / (1. + z)
        else:
            z = exp(x)
            t = (z - 1.) / (z + 1.)
        return t / (2. * x)

@cython.cdivision(True)
cdef inline Y_DTYPE_C _sigsigm(const Y_DTYPE_C x) nogil:
    """sigmoid(x) * (sigmoid(x) - 1)"""
    cdef:
        Y_DTYPE_C z

    if fabs(x) < 1e-10:
        return -0.25
    else:
        if x >= 0:
            z = exp(-x)
        else:
            z = exp(x)
    return -z / ((1. + z) ** 2)

cdef inline Y_DTYPE_C _sgn(const Y_DTYPE_C x) nogil:
    """sign(x)"""
    if x <= 0:
        return -1.
    else:
        return 1.
