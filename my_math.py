import random
import math

class Matrix:
    """Pure Python matrix implementation"""
    
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        if data is None:
            self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        else:
            self.data = data
    
    @staticmethod
    def from_list(lst):
        """Create matrix from 1D or 2D list"""
        if isinstance(lst[0], list):
            rows = len(lst)
            cols = len(lst[0])
            return Matrix(rows, cols, lst)
        else:
            return Matrix(len(lst), 1, [[x] for x in lst])
    
    def to_list(self):
        """Convert matrix to list"""
        if self.cols == 1:
            return [self.data[i][0] for i in range(self.rows)]
        return self.data
    
    def get(self, i, j):
        """Get element at position (i, j)"""
        return self.data[i][j]
    
    def set(self, i, j, value):
        """Set element at position (i, j)"""
        self.data[i][j] = value
    
    def copy(self):
        """Create a deep copy of matrix"""
        new_data = [[self.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(self.rows, self.cols, new_data)
    
    def reshape(self, new_rows, new_cols):
        """Reshape matrix"""
        if new_rows * new_cols != self.rows * self.cols:
            raise ValueError("Cannot reshape: size mismatch")
        
        flat = []
        for i in range(self.rows):
            for j in range(self.cols):
                flat.append(self.data[i][j])
        
        new_data = []
        idx = 0
        for i in range(new_rows):
            row = []
            for j in range(new_cols):
                row.append(flat[idx])
                idx += 1
            new_data.append(row)
        
        return Matrix(new_rows, new_cols, new_data)
    
    def add(self, other):
        """Element-wise addition"""
        if isinstance(other, (int, float)):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other
            return result
        
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")
        
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result
    
    def subtract(self, other):
        """Element-wise subtraction"""
        if isinstance(other, (int, float)):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - other
            return result
        
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match")
        
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        return result
    
    def multiply(self, other):
        """Element-wise multiplication"""
        if isinstance(other, (int, float)):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other
            return result
        
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match")
        
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * other.data[i][j]
        return result
    
    def dot(self, other):
        """Matrix multiplication (dot product)"""
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply: ({self.rows}x{self.cols}) · ({other.rows}x{other.cols})")
        
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                sum_val = 0.0
                for k in range(self.cols):
                    sum_val += self.data[i][k] * other.data[k][j]
                result.data[i][j] = sum_val
        return result
    
    def transpose(self):
        """Transpose matrix"""
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result
    
    def apply_function(self, func):
        """Apply function to each element"""
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = func(self.data[i][j])
        return result
    
    def sum_axis_0(self):
        """Sum along axis 0 (column-wise)"""
        result = Matrix(1, self.cols)
        for j in range(self.cols):
            sum_val = 0.0
            for i in range(self.rows):
                sum_val += self.data[i][j]
            result.data[0][j] = sum_val
        return result
    
    def sum_axis_1(self):
        """Sum along axis 1 (row-wise)"""
        result = Matrix(self.rows, 1)
        for i in range(self.rows):
            sum_val = 0.0
            for j in range(self.cols):
                sum_val += self.data[i][j]
            result.data[i][0] = sum_val
        return result
    
    def max_axis_1(self):
        """Max along axis 1 (row-wise)"""
        result = []
        for i in range(self.rows):
            max_val = self.data[i][0]
            for j in range(1, self.cols):
                if self.data[i][j] > max_val:
                    max_val = self.data[i][j]
            result.append(max_val)
        return result
    
    def argmax_axis_1(self):
        """Argmax along axis 1 (row-wise) - returns index of max value"""
        result = []
        for i in range(self.rows):
            max_idx = 0
            max_val = self.data[i][0]
            for j in range(1, self.cols):
                if self.data[i][j] > max_val:
                    max_val = self.data[i][j]
                    max_idx = j
            result.append(max_idx)
        return result
    
    def randomize_normal(self, mean=0.0, std=1.0):
        """Fill matrix with random normal distribution values"""
        for i in range(self.rows):
            for j in range(self.cols):
                # Box-Muller transform for normal distribution
                u1 = random.random()
                u2 = random.random()
                z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                self.data[i][j] = mean + std * z0
    
    def __str__(self):
        """String representation"""
        s = f"Matrix({self.rows}x{self.cols})[\n"
        for i in range(min(5, self.rows)):
            s += "  ["
            for j in range(min(5, self.cols)):
                s += f"{self.data[i][j]:8.4f}"
            if self.cols > 5:
                s += " ..."
            s += "]\n"
        if self.rows > 5:
            s += "  ...\n"
        s += "]"
        return s


# Activation functions
def relu(x):
    """ReLU activation"""
    return max(0.0, x)

def relu_derivative(x):
    """ReLU derivative"""
    return 1.0 if x > 0 else 0.0

def exp(x):
    """Exponential function with overflow protection"""
    try:
        return math.exp(x)
    except OverflowError:
        return math.exp(700)  # Large but not overflow

def log(x):
    """Logarithm with underflow protection"""
    return math.log(max(x, 1e-10))

def sqrt(x):
    """Square root"""
    return math.sqrt(x)


# Helper functions
def create_zeros(rows, cols):
    """Create zero matrix"""
    return Matrix(rows, cols)

def create_ones(rows, cols):
    """Create matrix filled with ones"""
    m = Matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            m.data[i][j] = 1.0
    return m

def create_random_normal(rows, cols, mean=0.0, std=1.0):
    """Create matrix with random normal values"""
    m = Matrix(rows, cols)
    m.randomize_normal(mean, std)
    return m


print("✓ Custom math library loaded successfully!")