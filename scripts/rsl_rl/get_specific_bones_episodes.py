import zarr    
                                                                                                                                                                                      
store = zarr.open_group('/move/data/bones/g1/zarr/locomotion_50hz.zarr', mode='r')
names = [str(n) for n in store['clip_names'][:]]
result = []                                 
for i, n in enumerate(names):                                                                                                                                                                        
    if 'slow' in n:                                                                                                                                                                   
        print(f'clip_id={i}: {n}')
        result.append(i)
# print(result)