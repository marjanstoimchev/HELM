from datamodules.utils import load_images
from data.hierarchy import process_hierarchy_config, extend_ys, extract_paths
from data.utils import prepare_train_test_validation, split_dataset_, random_sampling, DatasetFactory

# Dataset Pipeline Class
class DatasetPipeline:
    def __init__(self, yaml_path, test_size=0.2, val_size=0.1, seed=42, cache_dir='./Datasets/mlc_datasets_npy'):

        self.yaml_path = yaml_path
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.cache_dir = cache_dir
        self.df = None
        self.images = None
        self.labels = None
        # Process hierarchy configuration
        self.process_hierarchy_config_()
        

    def process_hierarchy_config_(self):
        h_config_out = process_hierarchy_config(self.yaml_path)

        self.h_config_out = h_config_out
        self.dataset_name = h_config_out.dataset_name
        self.num_classes = h_config_out.num_classes
        self.num_classes_extended = h_config_out.num_classes_extended
        self.label_names = h_config_out.label_names

        self.leaf_paths = h_config_out.leaf_paths.to_dict()
        self.intermediate_paths = h_config_out.intermediate_paths.to_dict()
        
        self.label_to_predecessors = h_config_out.label_to_predecessors.to_dict()
        # self.final_hierarchy_strings = h_config_out.final_hierarchy_strings.to_dict()
        self.final_hierarchy_strings = None
    
    def prepare_dataset_(self):

        df = DatasetFactory.create_dataset(self.dataset_name, self.num_classes).prepare_dataset()

        self.images, self.labels = load_images(
            df, 
            self.dataset_name, 
            image_size=224, 
            batch_size=32, 
            num_workers=16, 
            path = self.cache_dir,
            )
        
        print(self.images.max(), self.images.min())
        
        self.df = df


    def run_split(self, fraction=None):
        # Split dataset into train, validation, and test sets

        outputs = {}
        outputs_splits = prepare_train_test_validation(self.df, test_size=self.test_size, val_size=self.val_size, seed=self.seed)
        train_idx = outputs_splits['train']
        test_idx = outputs_splits['test']
        val_idx = outputs_splits['val']

        # ys_extended = extend_ys(self.labels, self.leaf_paths, self.intermediate_paths)

        # this is crucial #
        ordered_dict_items = [(k, dict(self.leaf_paths)[k]) for k in self.h_config_out.leaf_labels] # order the leaf paths by leaf label

        final_hierarchy_strings = {**dict(ordered_dict_items), **dict(self.intermediate_paths)}
        self.final_hierarchy_strings = final_hierarchy_strings


        # Call the function to extend ys
        ys_extended = extend_ys(
            self.labels,
            dict(ordered_dict_items), 
            final_hierarchy_strings,
            )
 
        # df_train = self.df.loc[train_idx]

        df_train = self.df.loc[train_idx].reset_index(drop=True)

        outputs['Y_tr'] = ys_extended[:, :self.num_classes][train_idx]
        outputs['Y_te'] = ys_extended[:, :self.num_classes][test_idx]
        outputs['Y_val'] = ys_extended[:, :self.num_classes][val_idx]

        outputs['Y_tr_h'] = ys_extended[train_idx]
        outputs['Y_te_h'] = ys_extended[test_idx]
        outputs['Y_val_h'] = ys_extended[val_idx]

        outputs['X_tr'] = self.images[train_idx]
        outputs['X_te'] = self.images[test_idx]
        outputs['X_val'] = self.images[val_idx]

        # Create unlabeled dataset if needed
        if fraction is not None:
            train_idx, unlabeled_idx = random_sampling(df_train, p=fraction, seed=self.seed)

            outputs['U'] = outputs['X_tr'][unlabeled_idx]
            outputs['Y_u'] = outputs['Y_tr'][unlabeled_idx]
            outputs['Y_u_h'] = outputs['Y_tr_h'][unlabeled_idx]
            
            print(f"UNLABELED (TRAIN) SET: {len(outputs['X_tr'][unlabeled_idx])}")
            outputs['X'] = (outputs['X_tr'][train_idx], outputs['Y_tr'][train_idx], outputs['Y_tr_h'][train_idx])
            print(f"LABELED (TRAIN) SET: {len(outputs['X_tr'][train_idx])}")
        else:
            outputs['X'] = (outputs['X_tr'], outputs['Y_tr'], outputs['Y_tr_h'])
            print(f"LABELED (TRAIN) SET: {len(outputs['X_tr'])}")
        
        print(f"VAL SET: {len(outputs['X_val'])}")
        print(f"TEST SET: {len(outputs['X_te'])}")
        del outputs['Y_tr'], outputs['Y_tr_h'], outputs['X_tr'], 

        return outputs
    
    def run_pipeline(self, fraction_labeled=None):
        # Orchestrate the entire pipeline
        self.prepare_dataset_()
        outputs = self.run_split(fraction_labeled)
        # Return the resulting data splits
        return outputs