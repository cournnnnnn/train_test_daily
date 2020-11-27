def ColorText(text, color):
    CEND      = '\033[0m'
    CBOLD     = '\033[1m'
    CRED    = '\033[91m'
    CGREEN  = '\033[32m'
    CYELLOW = '\033[33m'
    CBLUE   = '\033[34m'
    CVIOLET = '\033[35m'
    CBEIGE  = '\033[36m'
    if color == 'red':
        return CRED + CBOLD + text + CEND
    elif color == 'green':
        return CGREEN + CBOLD + text + CEND
    elif color == 'yellow':
        return CYELLOW + CBOLD + text + CEND
    elif color == 'blue':
        return CBLUE + CBOLD + text + CEND
    elif color == 'voilet':
        return CVIOLET + CBOLD + text + CEND
    elif color == 'beige':
        return CBEIGE + CBOLD + text + CEND

if __name__ == '__main__':
    text = ColorText('hello','beige')
    print(text)
