
import sys
sys.path.insert(0, sys.path[0]+'/scr')
sys.path.insert(0, sys.path[0]+'/scr/lib')
sys.path.insert(0, sys.path[0]+'/scr/load_data')


def main():

    import warnings
    warnings.filterwarnings("ignore")

    print("Start ... ")

    from main_pipeline import rm_mass_function_constrain
    rm_mass_function_constrain()

    print("... end. ")


if __name__ == '__main__':
    main()
