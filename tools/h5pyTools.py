#import ezodf
import numpy as np
import pdb
import h5py
class h5pyTools:
    def __init__(self):
            pass

    ############################################################
    # creates a new data-set if it doesn't exist, otherwise overwrites
    ############################################################
    def createOverwriteDS(self,grp,dsname,data,at=None):
        try:
            rec = grp.require_dataset(dsname, shape = np.shape(data), dtype = data.dtype, exact=False ) 
        # either different shape or not exisiting
        except TypeError:
            try:
                del grp[dsname]
            except KeyError:
                try:
                    grp.create_dataset(dsname,data=data,dtype = data.dtype)
                except TypeError:
                    string_dt = h5py.special_dtype(vlen=str)
                    data=np.array(data, dtype=string_dt)
                    grp.create_dataset(dsname, data=data, dtype=string_dt)
            else:
                grp.create_dataset(dsname,data=data,dtype = data.dtype)
        else:
            rec[:] = data
        # save attributes
        if at:
            #print('evaluating list shape')
            # print('list shape :', at, self.listShape(at))

            if len(self.listShape(at))==1:
                try:
                    grp[dsname].attrs[at[0]]=at[1]
                except:
                        grp[dsname].attrs[at[0]] = at[1]['.'][at[0]]

            else:
                for i in range(len(at)):
                    try:
                        grp[dsname].attrs[at[i][0]]=at[i][1]
                    except:
                        pdb.set_trace()


    ############################################################
    def getH5GroupName(self,f,groupNames):

        current_group = ''
        for i in range(len(groupNames)):
            if i == 0:
                try:
                    grpHandle = f.require_group(groupNames[i])
                except KeyError:
                    groupNames[i] = groupNames[i] + '_new'
                    grpHandle = f.require_group(groupNames[i])

            else:
                try:
                    grpHandle = f[current_group].require_group(groupNames[i])
                except KeyError:
                    groupNames[i] = groupNames[i] + '_new'
                    grpHandle = f[current_group].require_group(groupNames[i])

            current_group += groupNames[i] + '/'
        return (current_group[:-1],grpHandle)

    ############################################################
    def deleteElement(self,f,elementName):
        del f[elementName]

    ############################################################
    #def list_shape(self,inputList):
    def listShape(self,lst):
        def ishape(lst):
            shapes = [ishape(x) if isinstance(x, list) else [] for x in lst]
            shape = shapes[0]
            if shapes.count(shape) != len(shapes):
                raise ValueError('Ragged list')
            shape.append(len(lst))
            return shape

        return tuple(reversed(ishape(lst)))

