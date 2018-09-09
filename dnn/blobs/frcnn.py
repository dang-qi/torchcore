#import config
import numpy as np
#from config import dnn_cfg
from tools import bbox_tools
from .blobs import blobs

class frcnn_blobs( blobs ):
    def _build_category_labels( self, labels ):
        labels = labels.ravel()
        n = len( labels )
        clabels = np.zeros([ n,2 ] )
        for i,l in enumerate(labels) :
            clabels[i,int(l)] = 1.0
        return clabels.astype( np.float32 )

    def __init__( self, mcfg, dnn_cfg ):
        super().__init__( dnn_cfg )
        self._bbo = bbox_tools()
        self._mcfg = mcfg
        self._nclasses = mcfg['nclasses']

    def _add_rois_raw( self, images, blobs ):
        rois = []
        roilabels = []
        roibatches = []

        for image_ind, image in enumerate( images ):
            r = image.rois
            r_cls = image.roi_labels
            n = len( r )
            b = np.ones( [n,1], dtype=np.int32 ) * image_ind

            rois.append( r )
            roilabels.append( r_cls )
            roibatches.append( b )

        rois.append( blobs['gtboxes'] )
        roilabels.append( blobs['gtlabels'] )
        roibatches.append( blobs['gtbatches'] )

        rois = np.concatenate( rois )
        roilabels = np.concatenate( roilabels )
        roibatches = np.concatenate( roibatches )

        blobs['rois'] = rois
        blobs['roilabels'] = roilabels
        blobs['roibatches'] = roibatches

    def _do_stuff( self, images, blobs ):
        pass

    def _add_rois( self, images, blobs ):
        gtboxes = blobs['gtboxes']
        gtlabels = blobs['gtlabels']
        gtbatches = blobs['gtbatches']

        scales = blobs['scales']

        batch_rois = []
        batch_roi_labels = []
        batch_roi_targets = []
        batch_roi_batch_inds = []

        min_obj_side = self._dnn_cfg.TRAIN.MIN_OBJ_SIDE

        for image_ind,image in enumerate( images ):
            gt_indices = np.where( gtbatches == image_ind )[0]

            boxes = gtboxes[ gt_indices ]
            labels = gtlabels[ gt_indices ]
            rois = image.rois
            roi_cls = image.roi_labels
            image_cls = image.label - 1

            selection = np.where( roi_cls.ravel() == image_cls )[0]
            rois = rois[ selection ]
            roi_cls = roi_cls[ selection ]
            rois = self._remove_small_boxes( rois, min_obj_side )

            if self._dnn_cfg.FRCNN.TRAIN.ADD_GTBOXES :
                rois = np.concatenate( [rois,boxes] )

            nrois = len( rois )

            roi_labels = np.ones((nrois,self._nclasses)) * -1
            roi_targets = np.zeros((nrois,self._nclasses*4))

            ind0 = 4 * image_cls
            ind1 = 4 * (image_cls+1)

            if len( boxes ) > 0 :
                overlaps = self._bbo.overlap_gpu( rois, boxes )
                maxes = np.max( overlaps, axis=1 )
                argmaxes = np.argmax( overlaps, axis=1 )

                pos_I = np.where( maxes >= self._dnn_cfg.FRCNN.FG_MIN_OVERLAP )[0]
                neg_I = np.where( maxes < self._dnn_cfg.FRCNN.BG_MAX_OVERLAP )[0]

                if len( neg_I ) > 0 :
                    roi_labels[ neg_I, image_cls ] = 0
                if len( pos_I ) > 0 :
                    roi_labels[ pos_I, image_cls ] = 1
                    indices = np.arange(0,4) + 4*image_cls
                    roi_targets[ pos_I,ind0:ind1 ] = boxes[ argmaxes[pos_I] ]

            roi_indices = np.ones((nrois,1)) * image_ind
            rois = np.concatenate([roi_indices,rois],axis=1)

            rois_cls = np.zeros((nrois,self._nclasses*4))
            roi_batch_inds = np.ones((nrois,self._nclasses)) * -1
            rois_cls[:,ind0:ind1] = rois[:,1:]
            roi_batch_inds[:,image_cls] = rois[:,0]

            batch_rois.append( rois_cls )
            batch_roi_labels.append( roi_labels )
            batch_roi_targets.append( roi_targets )
            batch_roi_batch_inds.append( roi_batch_inds )

        batch_rois = np.concatenate( batch_rois )
        batch_roi_labels = np.concatenate( batch_roi_labels )
        batch_roi_targets = np.concatenate( batch_roi_targets )
        batch_roi_batch_inds = np.concatenate( batch_roi_batch_inds )

        blobs['clsrois'] = batch_rois.astype( np.float32 )
        blobs['labels'] = batch_roi_labels.astype( np.float32 )
        blobs['targets'] = batch_roi_targets.astype( np.float32 )
        #blobs['roibatches'] = batch_roi_batch_inds.astype( np.int32 )

        return blobs

    def _add_roi_deltas( self, images, blobs ):
        labels = blobs['roi_labels'].ravel()
        rois = blobs['rois']
        targets = blobs['roi_targets']
        pos_I = np.where( labels == 1.0 )[0]

        nrois = len(labels)

        deltas = np.zeros((nrois,4))
        mask = np.zeros((nrois,4))
        mask[pos_I] = 1.0

        if len( pos_I ) > 0 :
            deltas[ pos_I ] = self._bbo.transform( rois[ pos_I ], targets[ pos_I ] )

        blobs['roi_deltas'] = deltas.astype(np.float32)
        blobs['roi_mask'] = mask.astype(np.float32)

    def _prune_data( self, images, blobs ):
        labels = blobs['roi_labels']

        batch_size = self._mcfg['selection_batch_size']
        fg_ratio = self._dnn_cfg.FRCNN.TRAIN.FG_RATIO

        all_indices = []

        for i in range( self._nclasses):
            cls_labels = labels[:,i].ravel()

            pos_inds = np.where( cls_labels == 1 )[0]
            neg_inds = np.where( cls_labels == 0 )[0]

            npos = int(batch_size * fg_ratio)
            npos = np.min( [ len(pos_inds), npos ] )

            pos_selection = []

            if len(pos_inds) > 0 :
                pos_selection = np.random.choice( pos_inds, npos )

            nneg = batch_size - npos
            nneg = np.min( [ len(neg_inds), nneg ] )

            neg_selection = []

            if len(neg_inds) > 0 :
                neg_selection = np.random.choice( neg_inds, nneg )

            indices = np.concatenate( [ pos_selection, neg_selection ] ).astype( int )
            all_indices.append( indices )

        all_indices = np.concatenate( all_indices )

        blobs['rois'] = blobs['rois'][ all_indices ]
        blobs['roi_labels'] = blobs['roi_labels'][ all_indices ]
        blobs['roi_targets'] = blobs['roi_targets'][ all_indices ]
        blobs['roi_batch_inds'] = blobs['roi_batch_inds'][ all_indices ]

        rois = blobs['rois'].reshape([-1,4])
        roi_labels = blobs['roi_labels'].reshape([-1,1])
        roi_targets = blobs['roi_targets'].reshape([-1,4])
        roi_batch_inds = blobs['roi_batch_inds'].reshape([-1,1])

        selection = np.where( roi_labels.ravel() >= 0 )[0]

        rois = rois[ selection ]
        roi_labels = roi_labels[ selection ]
        roi_targets = roi_targets[ selection ]
        roi_batch_inds = roi_batch_inds[ selection ]

        blobs['roi_selection'] = selection.reshape([-1,1])
        blobs['rois'] = rois
        blobs['roi_labels'] = roi_labels
        blobs['roi_targets'] = roi_targets
        blobs['roi_batch_inds'] = roi_batch_inds

    def _convert_labels( self, image, blobs ):
        blobs['gtlabels'] = self._build_category_labels( blobs['gtlabels'] )
        print( blobs['gtlabels'].shape )

    def _convert_rois_to_tf( self, image, blobs ):
        height, width = blobs['data'].shape[1:3]

        feat_height = int( np.floor( height / self._dnn_cfg.COMMON.BASE_SIZE ) )
        feat_width = int( np.floor( width / self._dnn_cfg.COMMON.BASE_SIZE ) )

        batch_inds = blobs['roi_batch_inds'].ravel()
        rois = blobs['rois'] / self._dnn_cfg.COMMON.BASE_SIZE

        order = np.array( [ 1,0,3,2 ] )
        rois = rois[:,order]

        if feat_height == feat_width :
            rois = rois / ( feat_height - 1 )
        else :
            rois[:,0] = rois[:,0] / ( feat_height - 1 )
            rois[:,1] = rois[:,1] / ( feat_width - 1 )
            rois[:,2] = rois[:,2] / ( feat_height - 1 )
            rois[:,3] = rois[:,3] / ( feat_width - 1 )

        blobs['tf_rois'] = rois.astype( np.float32 )
        blobs['tf_roi_batch_inds'] = batch_inds.astype( np.int32 )

    def get_blobs( self, images ):
        blobs = {}

        self._add_data( images, blobs )
        self._add_gtboxes( images, blobs, remove_difficults=self._dnn_cfg.FRCNN.TRAIN.REMOVE_DIFFICULTS )
        self._add_rois_raw( images, blobs )
        #self._add_rois( images, blobs )
        #self._prune_data( images, blobs )
        #self._add_roi_deltas( images, blobs )
        #self._convert_labels( images, blobs )
        #self._convert_rois_to_tf( images, blobs )

        return blobs

    def get_debug_blobs( self, images ):
        blobs = {}

        self._add_data( images, blobs )
        self._add_gtboxes( images, blobs, remove_difficults=self._dnn_cfg.FRCNN.TRAIN.REMOVE_DIFFICULTS )
        self._add_rois( images, blobs )

        return blobs

class frcnn_blobs_hinge( frcnn_blobs ):
    def get_blobs( self, images ):
        blobs = {}

        self._add_data( images, blobs )
        self._add_gtboxes( images, blobs, remove_difficults=self._dnn_cfg.FRCNN.TRAIN.REMOVE_DIFFICULTS )
        #self._add_rois( images, blobs )
        #self._prune_data( images, blobs )
        #self._add_roi_deltas( images, blobs )
        self._convert_labels( images, blobs )
        #self._convert_rois_to_tf( images, blobs )

        return blobs

class frcnn_blobs_deploy( frcnn_blobs ):
    def _add_rois( self, image, blobs ):
        rois = image.rois
        roi_labels = image.roi_labels

        nrois = len( rois )
        roi_indices = np.zeros((nrois,1))

        blobs['rois'] = rois.astype( np.float32 )
        blobs['roi_labels'] = roi_labels.astype( np.int32 )
        blobs['roi_batch_inds'] = roi_indices.astype( np.int32 )
        return blobs

    def get_blobs( self, image ):
        blobs = {}

        self._add_data_deploy( image, blobs )
        self._add_rois( image, blobs )
        self._convert_rois_to_tf( image, blobs )

        return blobs

class system_blobs_deploy( frcnn_blobs ):
    def get_blobs( self, image ):
        blobs = {}

        self._add_data_deploy( image, blobs )
        #self._add_rois( image, blobs )
        #self._convert_rois_to_tf( image, blobs )

        return blobs

    def get_blobs_ndarray( self, image ):
        blobs = {}

        self._add_data_deploy_ndarray( image, blobs )

        return blobs
