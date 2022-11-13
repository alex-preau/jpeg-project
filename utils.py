import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from math import ceil,sqrt,cos,pi
import pylab
from scipy import fftpack
import matplotlib.cm as cm
from collections import Counter

#these are hardcoed for now, eventually will be compression-dependent
#so more compressed images will have larger values of the quanitzation table
QTY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  # luminance quantization table
                [12, 12, 14, 19, 26, 48, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

QTC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],  # chrominance quantization table
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])


#This is the order of a flattened 8x8 array read in 2d zig-zag order
zigzagOrder = np.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,
                           49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])

def padd(img_slice,size):
   # print(len(img_slice[0]))
    if len(img_slice) % size != 0:

        padd_size = (ceil(len(img_slice) / size ) * size ) - len(img_slice)

        img_slice = np.pad(img_slice, pad_width=((0, padd_size), (0, 0)))
    if len(img_slice[1]) % size != 0:

        padd_size = (ceil(len(img_slice[1]) / size ) * size ) - len(img_slice[1])

        img_slice = np.pad(img_slice, pad_width=((0, 0), (0, padd_size)))

    return img_slice

def block(img_slice, size):
    '''
    img slice is just an array of single pixel values
    size: length of blocks to be taken out
    '''
    #print(ceil(len(img_slice)/size))
    blocks = []
    padded = padd(img_slice,8)
    print(padded.shape)
    #padd 
    for i in range(ceil(len(padded)/size)):
        for j in range(ceil(len(padded[1])/size)):
            blocks.append(padded[i*size:i*size+8,j*size:j*size+8])
            #print(blocks[-1].shape)
            #print(i)
    return blocks
            
    
def dct(x, y, u, v, n):
    
    def alpha(a):
        if a==0:
            return sqrt(1.0/n)
        else:
            return sqrt(2.0/n)
    return alpha(u) * alpha(v) * cos(((2*x+1)*(u*pi))/(2*n)) * cos(((2*y+1)*(v*pi))/(2*n))

def getBasisImage(u, v, n):
    # for a given (u,v), make a DCT basis image
    basisImg = np.zeros((n,n))
    for y in range(0, n):
        for x in range(0, n):
            basisImg[y,x] = dct(x, y, u, v, n)
    return basisImg

def get_dcts():
    n = 8
    DCTs = []
    for u in range(0, n):
        for v in range(0, n):
            basisImg = getBasisImage(u, v, n)
            DCTs.append(basisImg)
    return DCTs

def calc_coef(block,DCTs):
    '''
    calculate all the coeficients for given block
    NOTE currenlty usses fft, change to calcualte in same manner as GPU
    '''
    #coefs =  np.zeros((len(DCTs),len(DCTs)))
    #for i in range(int(sqrt(len(DCTs)))):
    #    for j in range(int(sqrt(len(DCTs)))):
            #print(j)
    #        coef =np.dot( np.dot( DCTs[i*int(sqrt(len(DCTs))) + j].T  , block) , DCTs[i*int(sqrt(len(DCTs))) + j])
    #        print(coef)
    coefs = fftpack.dct(fftpack.dct(block, norm='ortho').T, norm='ortho')
    #print(coefs.shape)
    return coefs

def calc_all_coef(blocks):
    all_coef = []
    
    for b in blocks:
        #print(b)
        all_coef.append(calc_coef(b,None))
    return all_coef

def quantize(d_block,b_type):
    '''
    d_block: DCT'ed block 
    b_type: LUM or CHROM 
    
    '''
    if b_type == 'LUM':
        return np.rint(np.divide(d_block,QTY)).astype(int)
    else:
        return np.rint(np.divide(d_block,QTC)).astype(int)
    
def quantize_all(blocks,b_type):
    quantized = []
    for b in blocks:
        quantized.append(quantize(b,b_type))
    return quantized


def zig_block(block):
    '''
    converts a single block to a flat zig-zag-ed array
    '''
    #print(block)
    return block.flatten()[zigzagOrder]

def zig_zag_all(blocks):
    zagged = []
    for b in blocks:
        zagged.append(zig_block(b))
    return zagged


def DCT_quant(img_slice):
    '''
    This runs blocking, DCT, and quantization on an image slice
    '''
    blocks = block(img_slice,8)
    dtf_coef = calc_all_coef(blocks)
    quantized = quantize_all(dtf_coef,'LUM')
    return quantized

def trim(array):
    """
    this removes trailing 0s from the sequence
    returns single elem seq if all 0s
    """
    trimmed = np.trim_zeros(array, 'b')
    if len(trimmed) == 0:
        trimmed = np.zeros(1)
    return trimmed

def run_length_encoding(zagged):
    """
    run length encoding is tricky, is based on difference between current and previous element and is interconnected
    between blocks
    NOTE this means its v difficult (impossible?) to parallelize
    
    format for DC components is (size)(amplitude)
    format for AC components is (run_length, size) (Amplitude of non-zero)
    takes zagged array of all blocks
    returns run length encoded values as an array of tuples
    """
    encoded = []
    curr_run = 0
    eob = ("EOB",) #end of block

    for i in range(len(zagged)):
        for j in range(len(zagged[i])):
            trimmed = trim(zagged[i])
            if j == len(trimmed):
                encoded.append(eob)  # EOB
                break
            if i == 0 and j == 0:  # just encode value of this (first elem first block)
                encoded.append((int(trimmed[j]).bit_length(), trimmed[j]))
            elif j == 0:  # encode the dif between first value this block and first of prev block
                diff = int(zagged[i][j] - zagged[i - 1][j])
                if diff != 0:
                    encoded.append((diff.bit_length(), diff))
                else:
                    encoded.append((1, diff))
                curr_run = 0
            elif trimmed[j] == 0:  # increment run_length by one in case of a zero
                curr_run += 1
            else:  
                encoded.append((curr_run, int(trimmed[j]).bit_length(), trimmed[j]))
                curr_run = 0
            # end of block
        if not (encoded[len(encoded) - 1] == eob):
            encoded.append(eob)
    return encoded

def lowest_prob_pair(p):
    # Return pair of symbols from run length encoded seq with lowest probabilities
    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]

def get_freq_dict(array):
    """
    returns a dict where the keys are the values of the array, and the values are their frequencies
    :param numpy.ndarray array: intermediary stream as array
    :return: frequency table
    """
    #
    data = Counter(array)
    result = {k: d / len(array) for k, d in data.items()}
    return result


def find_huffman(p):
    """
    p is frequency array, computed above
    returns a Huffman table for an sequence with distribution p
    
    
    """
    # Base case of only two symbols, assign 0 or 1 arbitrarily; frequency does not matter
    if len(p) == 2:
        return dict(zip(p.keys(), ['0', '1']))

    # Create a new distribution by merging lowest probable pair
    next_p = p.copy()
    a1, a2 = lowest_prob_pair(p)
    p1, p2 = next_p.pop(a1), next_p.pop(a2)
    next_p[a1 + a2] = p1 + p2

    # Recurse and construct code on new distribution
    huff_table = find_huffman(next_p)
    huff_value_a1a2 = huff_table.pop(a1 + a2)
    huff_table[a1], huff_table[a2] = huff_value_a1a2 + '0', huff_value_a1a2 + '1'

    return huff_table


# test functions


def reconstruct_slice(quantized,orig_slice):
    '''
    takes the quantized slice and returns it
    an easy way to test if the DCT + quantization is working'''
    new_img = np.zeros((len(padd(orig_slice,8)),len(padd(orig_slice,8)[1])))

    for idx,qb in enumerate(quantized):
        i = int(idx / (len(padd(orig_slice,8)[1])//8))
        j = idx % (len(padd(orig_slice,8)[1])//8)
       # print(j)
        #print(idx - (i*(len(padd(image_r,8)[1])//8)))

        coefs = fftpack.idct(fftpack.idct(np.multiply(qb,QTY), norm='ortho').T, norm='ortho').astype(int)
        new_img[i*8:i*8+8,j*8:j*8+8] = coefs
    return new_img


