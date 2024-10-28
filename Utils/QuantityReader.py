def read_quantity(quantity, paramsfile):

    """
    Takes in a quantity key for the parameters and returns the value.

    Args:
        quantity (str): The quantity desired, e.g. I1 (which is first order exchange stiffness)
        paramsfiles (str): Path to the file `MicromagneticParams`, in which these data are stored

    Returns:
        The corresponding quantity read from the file

    """

    with open(paramsfile, 'r') as f:
        for line in f.readlines():
            if line.split('=')[0] == quantity:
                return float(line.split('=')[1])
    
    raise ValueError('Invalid quantity key given:', quantity)
