from .Informer import Model as Informer_Model
from .Transformer import Model as Transformer_Model
from .Autoformer import Model as Autoformer_Model
from .TimesNet import Model as TimesNet_Model
from .Nonstationary_Transformer import Model as Nonstationary_Transformer_Model
from .DLinear import Model as DLinear_Model
from .FEDformer import Model as FEDformer_Model
from .Reformer import Model as Reformer_Model
from .PatchTST import Model as PatchTST_Model
from .Crossformer import Model as Crossformer_Model
from .iTransformer import Model as iTransformer_Model
from .Koopa import Model as Koopa_Model
from .FiLM import Model as FiLM_Model
from .FreTS import Model as FreTS_Model
from .TimeMixer import Model as TimeMixer_Model
from .TiDE import Model as TiDE_Model
from .TSMixer import Model as TSMixer_Model

models = {
    'Informer': Informer_Model,
    'Transformer': Transformer_Model,
    'Autoformer': Autoformer_Model,
    'TimesNet': TimesNet_Model,
    'Nonstationary_Transformer': Nonstationary_Transformer_Model,
    'DLinear': DLinear_Model,
    'FEDformer': FEDformer_Model,
    'Reformer': Reformer_Model,
    'PatchTST': PatchTST_Model,
    'Crossformer': Crossformer_Model,
    'FiLM': FiLM_Model,
    'iTransformer': iTransformer_Model,
    'Koopa': Koopa_Model,
    'TiDE': TiDE_Model,
    'FreTS': FreTS_Model,
    'TimeMixer': TimeMixer_Model,
    'TSMixer': TSMixer_Model,
}
