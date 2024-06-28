import numpy as np
import torch
from types import SimpleNamespace
    

def geometric_median_array(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:`n`, whose elements are each a ``torch.Tensor`` of shape ``(d,)``
    :param weights: ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a ``torch.Tensor`` object of shape :math:``(d,)``
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list.
    """
    with torch.no_grad():
        # initialize median estimate at mean
        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)
        logs = [objective_value]

        # Weiszfeld iterations
        early_termination = False
        for _ in range(maxiter):
            prev_obj_value = objective_value
            norms = torch.stack([torch.norm((p - median).view(-1)) for p in points])
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)

            objective_value = geometric_median_objective(median, points, weights)
            logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break

    median = weighted_average(points, new_weights)  # allow autodiff to track it
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination="function value converged within tolerance" if early_termination else "maximum iterations reached",
        logs=logs,
    )

def geometric_median_tensor(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:`n`, whose elements are each a ``torch.Tensor`` of shape ``(d,)``
    :param weights: ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a ``torch.Tensor`` object of shape :math:``(d,)``
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list.
    """
    with torch.no_grad():
        # initialize median estimate at mean
        new_weights = weights
        median = weighted_average_tensor(points, weights)
        objective_value = geometric_median_objective_tensor(median, points, weights)
        logs = [objective_value]

        # Weiszfeld iterations
        early_termination = False
        for _ in range(maxiter):
            prev_obj_value = objective_value
            norms = torch.norm(points - median, dim=1)
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average_tensor(points, new_weights)

            objective_value = geometric_median_objective_tensor(median, points, weights)
            logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break

    median = weighted_average_tensor(points, new_weights)  # allow autodiff to track it
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination=0 if early_termination else -1,
        logs=logs,
    )


def weighted_average_tensor(points, weights):
    weights = weights / torch.sum(weights)
    ret = (points.T * weights).T
    ret = torch.sum(ret, dim=0)
    
    return ret

@torch.no_grad()
def geometric_median_objective_tensor(median, points, weights):
    
    out = torch.norm(points - median, dim=1)
    out = torch.sum(out*weights) / torch.sum(weights)
    return out



def geometric_median_per_component(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:``n``, where each element is itself a list of ``numpy.ndarray``.
        Each inner list has the same "shape".
    :param weights: ``numpy.ndarray`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a list of ``numpy.ndarray`` of the same "shape" as the input.
        - `termination`: string explaining how the algorithm terminated, one for each component. 
        - `logs`: function values encountered through the course of the algorithm.
    """
    components = list(zip(*points))
    median = []
    termination = []
    logs = []
    new_weights = []
    for component in components:
        ret = geometric_median_array(component, weights, eps, maxiter, ftol)
        median.append(ret.median)
        new_weights.append(ret.new_weights)
        termination.append(ret.termination)
        logs.append(ret.logs)
    return SimpleNamespace(median=median, termination=termination, logs=logs)

def weighted_average(points, weights):
    weights = weights / weights.sum()
    ret = points[0] * weights[0]
    for i in range(1, len(points)):
        ret += points[i] * weights[i]
    return ret

@torch.no_grad()
def geometric_median_objective(median, points, weights):
    return np.average([torch.norm((p - median).reshape(-1)).item() for p in points], weights=weights.cpu())



if __name__ == '__main__':
    
    points = torch.randn(50, 2048)
    weights = torch.ones(50)
    
    temp_points = [p for p in points] 
    #orig_median = weighted_average(temp_points, weights)
    #orig_objective_value= geometric_median_objective(orig_median, temp_points, weights)
    #
    #norms = torch.stack([torch.norm((p - orig_median).view(-1)) for p in temp_points])
    #
    #
    #median = weighted_average_tensor(points, weights)
    #objective = geometric_median_objective_tensor(median, points, weights)
    #
    #print('median difference: {}'.format(torch.norm(orig_median - median)))
    #print('objective difference: {}'.format(torch.norm(orig_objective_value - objective)))
    
    out_orig = geometric_median_array(temp_points, weights, ftol=1e-20)
    out = geometric_median_tensor(points, weights, ftol=1e-20)
    
    t1 = out_orig.median
    t2 = out.median
    print(torch.norm(t1 - t2))
    x = 5