import numpy as np
import scipy.signal
#import nmrglue as ng
import networkx as nx
from numpy import corrcoef
import matplotlib.pyplot as plt

#以下函数大部分来自nuscon在nmrbox中的python包，只是稍有调整

class PeakTable:
    def __init__(self, peaks, intensities=None):
        """
        :param peaks: An array of peak positions.
        :param intensities: An optional array of peak intensities.
        """
        self.peak_positions = np.array(peaks)
        
        # 检查是否也提供了强度，并确保它与峰值位置的数量相匹配
        if intensities is not None:
            if len(intensities) != len(peaks):
                raise ValueError("Peaks and intensities must have the same length.")
            self.intensities = np.array(intensities)
        else:
            self.intensities = None

    def get_par(self, param):
        """
        根据参数名称返回相应的数据。
        :param param: 请求的参数名，如 '*_HZ' 或 'HEIGHT'。
        :return: 请求的参数数组。
        """
        if param == '*_HZ':
            return self.peak_positions[:, np.newaxis]  # 转换为二维数组
        elif param == 'HEIGHT' and self.intensities is not None:
            return self.intensities[:, np.newaxis]
        else:
            raise ValueError(f"Unsupported parameter: {param}")

# 使用 scipy.signal.find_peaks 获取峰值位置的函数
def find_peaks(signal, height=None, distance=None, prominence=0.01):
    peaks, _ = scipy.signal.find_peaks(signal, height=height, distance=distance, prominence=prominence)
    return peaks

'''
#使用nmrglue来获取谱峰位置。暂时放弃，只能设置高度阈值，无法调整别的参数，效果不佳
def find_peaks_nmr(data, threshold):
    """
    使用nmrglue寻找NMR谱中的峰值，并返回它们在数据中的索引。

    :param data: NMR数据数组。
    :param threshold: 检测峰值的阈值。
    :return: 峰值索引的列表。
    """
    # 使用nmrglue找到峰值
    peaks = ng.analysis.peakpick.pick(data, pthres=threshold,algorithm="connected")

    # 从nmrglue返回的结果中提取峰值索引
    peak_indices = [peak[0] for peak in peaks]  
    peak_indices = np.array(peak_indices)
    peak_indices = peak_indices.astype(int)
    return peak_indices
'''

def shape2(A, dims=2, axis=None):
    """
    The numpy shape function returns a singleton tuple for 1D arrays.
    This function fills in 1s out to a minimum number of desired dimensions.
    This function allows length of specific axes (or all) to be returned.
    Check for this and return row and column counts.
    :param A: array
    :param dims: minimum number of dimensions to report size on
    :param axis: indices of dimensions to report
    :return: int (for single axis), list of ints (for multiple)
    """
    # column vector returns shape as: (num,)
    # this 1-D tuple fails to provide a 1 as the second entry
    # => catch that case here

    try:
        shape = np.shape(A)
    except TypeError:
        raise TypeError('can not get shape of input')

    # extend shape with 1s to reach the desired minimum dimensionality
    while len(shape) < dims:
        shape += (1,)

    # create a list of dimension indices, return the size of each
    if axis is None:
        axis = list(range(len(shape)))
    elif isinstance(axis, int):
        axis = [axis]

    try:
        out = [shape[a] for a in axis]
    except IndexError:
        raise IndexError('axis value out of range')

    if len(out) == 1:
        return out[0]
    else:
        return out

def pairwise_distance(A, B, cutoff=None, weights=None):
    """
    Compute distance between all positions of A and B
    :param A: matrix where row vectors are positions
    :param B: matrix where row vectors are positions
    :param cutoff: maximum distance allowed
    :param weights: weighting factors for each dimension used for norm
    :return: matrix of distances, rows = A objects, cols = B objects
    """
    # A and B are numpy arrays.  Each row is the multi-dim position of a peak.
    numA, dimA = shape2(A)
    numB, dimB = shape2(B)

    if weights is None:
        weights = np.ones(dimA)

    if not (dimA == dimB == len(weights)):
        raise ValueError('dimensionality error')

    D = np.empty([numA, numB])
    for i_a in range(numA):
        a = A[i_a, :]
        for i_b in range(numB):
            b = B[i_b, :]

            # weighted difference in positions between a and b
            w_diff = np.multiply(a-b, weights)
            D[i_a, i_b] = np.linalg.norm(w_diff)

    # make sure no distances are larger than maximum allowed cutoff
    if cutoff is not None:
        D = np.minimum(D, cutoff)

    return D

def max_matching_size(positionsA, positionsB, weights=None, cutoff=None, show_plot=False):
    """
    Compute cardinality of a maximum matching on the bipartite graph where
    edges are based on distances between positions from 2 sets.
    :param positionsA: numpy array, rows are position vectors
    :param positionsB: numpy array, rows are position vectors
    :param weights: vector of weighting factors used to compute distance
    :param cutoff: cutoff distance for considering positions connected
    :param show_plot: show a plot of the graph
    :return:
    """
    # create a mask of 1's and 0s indicating which peak pairs are within cutoff
    D = pairwise_distance(
        positionsA,
        positionsB,
        weights=weights,
        cutoff=cutoff)

    # find all inject : recover peak pairs within cutoff distance
    # P is a tuple of 2 arrays where item i from first array is paired with item i from second
    P = np.where(D < cutoff)

    # need unique names for the graph vertices => prepend "I" and "R"
    I = ['I' + str(p) for p in P[0]]
    R = ['R' + str(p) for p in P[1]]

    IR_pairs = zip(I, R)

    G = nx.Graph()
    G.add_nodes_from(I, bipartite=0)
    G.add_nodes_from(R, bipartite=1)
    G.add_edges_from(list(IR_pairs))
    if show_plot:
        nx.draw(G, with_labels=True)

    if not nx.is_bipartite(G):
        raise ValueError('graph is not bipartite')

    # maximum matching does not work on disconnected graph
    # => iterate through each connected subgraph and tally the maximum matchings
    # graphs = list(nx.connected_component_subgraphs(G))  # <- deprecated
    graphs = [G.subgraph(c) for c in nx.connected_components(G)]
    total_edges_in_max_matching = 0
    for g in graphs:
        # maximum matching seems to be directional, so each edge is counted twice
        MM = nx.bipartite.maximum_matching(g)
        total_edges_in_max_matching += int(len(MM) / 2)

    return total_edges_in_max_matching

def hausdorff(A, B, cutoff=None, weights=None):
    """
    Symmetric Hausdorff (distance between two sets) using maximum pairwise distance and axis weights
    :param A: numpy array where rows are object positions
    :param B: numpy array where rows are object positions
    :param cutoff: maximum distance allowed from a to b
    :param weights: weighting factors for each dimension (columns of A,B) used for distance computation
    :return:
    """
    numA = shape2(A, axis=0)
    numB = shape2(B, axis=0)

    D = pairwise_distance(
        A, B,
        weights=weights,
        cutoff=cutoff)

    # minimum values along axis=0 (A) -> for each b, what is distance to closest a
    minBA = np.amin(D, axis=0)
    # minimum values along axis=1 (B) -> for each a, what is distance to closest b
    minAB = np.amin(D, axis=1)

    H_AB = np.sqrt((1 / numA) * np.sum(minAB ** 2))
    H_BA = np.sqrt((1 / numB) * np.sum(minBA ** 2))

    return (H_AB + H_BA)/2

def find_nearest_intensity(A, B, A_I, B_I, distance_limit):
    B_matched_intensities = [0] * len(A)  # 初始化匹配强度列表
    B_matched_positions = [None] * len(A)  # 初始化匹配位置列表
    matched_B_indices = set()  # 跟踪已匹配的B中的峰

    # 根据A中的强度排序A及其位置
    sorted_indices = np.argsort(A_I)[::-1]  # 降序排序
    A_sorted = np.array(A)[sorted_indices]
    A_I_sorted = np.array(A_I)[sorted_indices]

    # 遍历排序后的A中的每个位置
    for index, a_pos in enumerate(A_sorted):
        min_distance = float('inf')
        matched_intensity = 0
        matched_position = None

        # 遍历B中的每个位置
        for b_index, b_pos in enumerate(B):
            if b_index in matched_B_indices:
                continue  # 如果B中的这个峰已经匹配过，跳过

            distance = abs(a_pos - b_pos)
            if distance < min_distance and distance <= distance_limit:
                min_distance = distance
                matched_intensity = B_I[b_index]
                matched_position = b_pos
                matched_B_index = b_index  # 记录匹配的B中峰的索引

        if matched_position is not None:
            matched_B_indices.add(matched_B_index)  # 标记已匹配

        B_matched_intensities[index] = matched_intensity
        B_matched_positions[index] = matched_position

    # 重新按照A的原始顺序排列匹配结果
    original_order_indices = np.argsort(sorted_indices)
    B_matched_intensities = np.array(B_matched_intensities)[original_order_indices].tolist()
    B_matched_positions = np.array(B_matched_positions)[original_order_indices].tolist()

    return B_matched_intensities, B_matched_positions

    # 打印A的位置及其对应的B的位置
    '''print("Matched positions:")
    for a_pos, b_pos in zip(A, B_matched_positions):
        print(f"A position: {a_pos} matched with B position: {b_pos}")

    # 打印A的强度及其对应的B的强度
    print("\nMatched intensities:")
    for a_intensity, b_intensity in zip(A_I, B_matched_intensities):
        print(f"A intensity: {a_intensity} matched with B intensity: {b_intensity}")'''

def m1(recover_peakList: PeakTable, inject_peakList: PeakTable, cutoff=None, weights=None):
    """
    Symmetric Hausdorff metric using a maximum distance cutoff and axis weightings.
    :param recover_peakList: peak list recovered from processing
    :param inject_peakList: peak list injected
    :param cutoff: maximum distance allowed for a peak to be considered as recovered
    :return: score
    """
    inject_HZ = inject_peakList.get_par('*_HZ')
    recover_HZ = recover_peakList.get_par('*_HZ')

    H = hausdorff(A=inject_HZ, B=recover_HZ, cutoff=cutoff, weights=weights)

    return 1 - (H / cutoff) if cutoff is not None else 1 - H

def m2(recover_peakList: PeakTable, inject_peakList: PeakTable, distance_limit,use_abs=False):
    """
    Linearity of the recovered peak intensity R-squared
    :param recover_peakList: peak list recovered from processing
    :param inject_peakList: peak list injected
    :return:score,matched_positions
    """
    #
    # get peak positions in PPM (data as numpy arrays)
    inject_HEIGHT = inject_peakList.get_par('HEIGHT')
    recover_HEIGHT = recover_peakList.get_par('HEIGHT')
    inject_HZ = inject_peakList.get_par('*_HZ')
    recover_HZ = recover_peakList.get_par('*_HZ')
    
    if use_abs:
         inject_HEIGHT = np.abs(inject_HEIGHT)
         recover_HEIGHT = np.abs(recover_HEIGHT)

    # correlation function returns num objects x num objects matrix
    # => the topr right (or lower left) entries given correlation coeff between 2 inputs
    # ==> diagonal entries are self correlations (should be 1)
    recover_HEIGHT,matched_positions=find_nearest_intensity(inject_HZ.flatten(),recover_HZ.flatten(),inject_HEIGHT.flatten(),recover_HEIGHT.flatten(),distance_limit)
    score = np.corrcoef(inject_HEIGHT.flatten(), recover_HEIGHT)[0, 1]
    score=score*score
    return score,matched_positions

def m3(recover_peakList: PeakTable, inject_peakList: PeakTable, cutoff):
    """
    Metric based on the percentage of injected peaks that are recovered within a specified cutoff distance.
    :param recover_peakList: peak list recovered from processing
    :param inject_peakList: peak list injected
    :param cutoff: maximum distance allowed for a peak to be considered as recovered
    :return: the percentage of injected peaks that have a matched recovered peak
    """

    # Get peak positions in HZ (data as numpy arrays)
    inject_HZ = inject_peakList.get_par('*_HZ')
    recover_HZ = recover_peakList.get_par('*_HZ')

    # Provide a count for how many of the injected peaks have a unique recovered peak within the cutoff distance
    total_edges_in_max_matching = max_matching_size(
        positionsA=inject_HZ,
        positionsB=recover_HZ,
        cutoff=cutoff)

    # Calculate the percentage of injected peaks that are matched
    percentage_recovered = total_edges_in_max_matching / len(inject_HZ)

    return percentage_recovered

def m4(recover_peakList: PeakTable, inject_peakList: PeakTable, cutoff):
    """
    Metric based on percentage of recovered peaks that are false.
    :param recover_peakList: peak list recovered from processing
    :param inject_peakList: peak list injected
    :param cutoff: maximum distance allowed for a peak to be considered as recovered
    :return:
    """

    # get peak positions in HZ (data as numpy arrays)
    try:
        inject_HZ = inject_peakList.get_par('*_HZ')
    except ValueError:
        raise ValueError('unable to determine injected peak positions in Hz')
    try:
        recover_HZ = recover_peakList.get_par('*_HZ')
    except ValueError:
        raise ValueError('unable to determine recovered peak positions in Hz')

    # this will provide a count for how many of the injected peaks have
    # a unique recovered peak within the cutoff distance
    total_edges_in_max_matching = max_matching_size(
        positionsA=inject_HZ,
        positionsB=recover_HZ,
        cutoff=cutoff)

    # all peaks in recover_peakList that are not in matching are taken as false positives
    # the ratio reported below -> 0 as FP grows and -> 1 as FP is eliminated
    return total_edges_in_max_matching / len(recover_HZ)

def m5(recover_signal, reference_signal: PeakTable, threshold):
    """
    Metric based on the sparsity of the spectral.
    """
    recover_signal=np.abs(recover_signal)
    reference_signal=np.abs(reference_signal)
    recover_signal[recover_signal<threshold]=0
    reference_signal[reference_signal<threshold]=0
    reference_sparsity=np.count_nonzero(reference_signal)/len(reference_signal)
    recover_sparsity=np.count_nonzero(recover_signal)/len(recover_signal)
    
    return (recover_sparsity-reference_sparsity)/reference_sparsity


# Define the function to evaluate signals
def evaluate_signals(fs,Nf,f0,ref_f0,reference_signal_matrix, recovered_signal_matrix,height_ref=0.02,height_rec=0.035, distance=5, prominence=0.01,cutoff=0.25,distance_limit=0.05,threshold=0.0001,use_abs=False):
    # 检查输入数据是一维还是二维，并相应地调整它们
    if np.ndim(reference_signal_matrix) == 1:
        # 如果输入是一维，将其扩展为二维（1 x N）
        reference_signal_matrix = np.expand_dims(reference_signal_matrix, axis=0)
        recovered_signal_matrix = np.expand_dims(recovered_signal_matrix, axis=0)
    
    # 确保reference_signal_matrix和recovered_signal_matrix具有相同的行数
    assert reference_signal_matrix.shape[0] == recovered_signal_matrix.shape[0], "Reference and recovered signals must have the same number of rows."
    results = []  # 存储每一行的评估结果
    reference_peaks_matrix=[]
    reference_intensities_matrix=[]
    recovered_peaks_matrix=[]
    recovered_intensities_matrix=[]
    
    for i in range(reference_signal_matrix.shape[0]):
        # 逐行处理
        reference_signal = reference_signal_matrix[i, :]
        recovered_signal = recovered_signal_matrix[i, :]
        #normalized
        reference_signal = reference_signal / np.abs(reference_signal).max()
        recovered_signal = recovered_signal / np.abs(recovered_signal).max()
        # Find peaks
        reference_peaks = find_peaks(reference_signal, height_ref, distance,prominence)
        recovered_peaks = find_peaks(recovered_signal, height_rec, distance,prominence)
        #reference_peaks = find_peaks_nmr(reference_signal, height_ref)
        #recovered_peaks = find_peaks_nmr(recovered_signal, height_rec)
        ff = fs/Nf 
        # Get intensities at the peaks
        reference_intensities = reference_signal[reference_peaks]
        recovered_intensities = recovered_signal[recovered_peaks]
        #ppm= (ff-(ref_f0-f0)*1e6)/f0
        reference_peaks = ((ff*reference_peaks-0.5*fs)-(ref_f0-f0)*1e6)/f0 #varian format
        recovered_peaks = ((ff*recovered_peaks-0.5*fs)-(ref_f0-f0)*1e6)/f0 #varian format
        # Create PeakTable instances
        reference_peak_table = PeakTable(reference_peaks, reference_intensities)
        recovered_peak_table = PeakTable(recovered_peaks, recovered_intensities)

        # Calculate scores
        score_m1 = m1(recovered_peak_table, reference_peak_table, cutoff, weights=None)
        score_m2,matched_positions = m2(recovered_peak_table, reference_peak_table,distance_limit,use_abs)
        score_m3 = m3(recovered_peak_table, reference_peak_table, cutoff)
        score_m4 = m4(recovered_peak_table, reference_peak_table, cutoff)
        score_m5 = m5(recovered_signal, reference_signal, threshold)
        # 存储结果
        results.append((score_m1, score_m2, score_m3, score_m4, score_m5,len(reference_peaks), len(recovered_peaks)))
        reference_peaks_matrix.append(reference_peaks)
        reference_intensities_matrix.append(reference_intensities)
        recovered_peaks_matrix.append(recovered_peaks)
        recovered_intensities_matrix.append(recovered_intensities)
        #print(f" frequency accuracy m1: {score_m1}, linearity of peak intensity m2:{score_m2},true positive rate m3: {score_m3},false positive rate m4: {score_m4}")

    return results,reference_peaks_matrix,reference_intensities_matrix,recovered_peaks_matrix,recovered_intensities_matrix,matched_positions

if __name__ == "__main__":
    pass