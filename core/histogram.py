import cv2
from matplotlib.figure import Figure

class Histogram():
    
    @staticmethod
    def computeHistoColored(image): #BGR order
        histB = cv2.calcHist([image],[0],None,[256],[0,256]).flatten()
        histG = cv2.calcHist([image],[1],None,[256],[0,256]).flatten()
        histR = cv2.calcHist([image],[2],None,[256],[0,256]).flatten()
        return histB, histG, histR

    @staticmethod
    def computeHistoGray(gray_image):
        hist= cv2.calcHist([gray_image],[0],None,[256],[0,256]).flatten()
        return hist



    @staticmethod
    def plot_colored_histogram(histB, histG, histR):
        """Create matplotlib figure for colored histogram"""
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        x = range(256)
        ax.bar(x, histB, color='blue', alpha=0.3, label='Blue')
        ax.bar(x, histG, color='green', alpha=0.3, label='Green')
        ax.bar(x, histR, color='red', alpha=0.3, label='Red')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('RGB Histogram')
        ax.legend()
        ax.grid(True)
        return fig
    
    @staticmethod
    def plot_gray_histogram(hist):
        """Create matplotlib figure for grayscale histogram"""
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.bar(range(256),hist, color='black')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Grayscale Histogram')
        ax.grid(True)
        return fig
    

    @staticmethod
    def compute_cdf_colored(histB, histG, histR):
        """Compute CDF for each color channel"""
        cdfB = histB.cumsum()
        cdfB_normalized = cdfB / cdfB.max()
        
        cdfG = histG.cumsum()
        cdfG_normalized = cdfG / cdfG.max()
        
        cdfR = histR.cumsum()
        cdfR_normalized = cdfR / cdfR.max()
        
        return cdfB_normalized, cdfG_normalized, cdfR_normalized

    @staticmethod
    def compute_cdf_gray(hist):
        """Compute CDF for grayscale"""
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        return cdf_normalized

    @staticmethod
    def plot_cdf_colored(cdfB, cdfG, cdfR, show_blue=True, show_green=True, show_red=True):
        """Plot CDF with selectable color channels"""
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        
        if show_blue:
            ax.plot(cdfB, color='blue', label='Blue')
        if show_green:
            ax.plot(cdfG, color='green', label='Green')
        if show_red:
            ax.plot(cdfR, color='red', label='Red')
        
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('RGB CDF')
        ax.legend()
        ax.grid(True)
        return fig

    @staticmethod
    def plot_cdf_gray(cdf):
        """Plot grayscale CDF"""
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(cdf, color='black')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Grayscale CDF')
        ax.grid(True)
        return fig




