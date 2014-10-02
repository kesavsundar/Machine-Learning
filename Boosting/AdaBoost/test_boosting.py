__author__ = 'kesavsundar'
import Boosting
import sys
sys.path.append('/home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import tenfold

if __name__ == '__main__':
    ada_boost = Boosting()
    ada_boost.init_boosting_model()
    tf = tenfold.Tenfold(ada_boost.xy_data, ada_boost)
    tf.inbuilt_tenfold_train()