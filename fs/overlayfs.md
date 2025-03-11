OverlayFS (Overlay Filesystem) is a union filesystem used in Linux to merge multiple directories so that they appear as a single directory. 
Used in: containers (Docker), LiveCDs, snapshot systems

## Basic Concepts 
OverlayFS overlays two directories (called "layers") together:
- Upper Layer: The writable layer where all new files or modifications are stored.
- Lower Layer: The read-only layer that provides the base data and cannot be modified.

When you modify a file:
- If the file originally exists in the Lower Layer, it will be copied to the Upper Layer (Copy-on-Write, CoW) and modified there.
- If the file already exists in the Upper Layer, it will be modified directly.

## Key Features
- Whiteout
    A special character device file in the Upper Layer that hides a file from the Lower Layer. (for delete upper layer files)
- Opaque Directories
    A marker indicating that a directory in the Upper Layer completely overrides its counterpart in the Lower Layer. (for update upper layer files)
- Copy on Write
    When a user attempts to modify a file in the Lower Layer, the file is copied to the Upper Layer.
    The user modifies the copied version in the Upper Layer, while the Lower Layer remains unchanged.
<!-- - Multi-Layer Storage
    OverlayFS supports multiple Lower Layers for layered storage architecture:
    `lowerdir=base-layer:patch-layer:fix-layer`
    base-layer → The bottom-most layer (read-only).
    patch-layer → Provides additional patches (read-only).
    fix-layer → Fixes and updates (read-only).
    upperdir → The user-writable layer. -->


## Example
Assume you have two directories:
- /upperdir: The read-only lower layer.
- /lowerdir: The writable upper layer.
- /merged: The final merged mount point.
To create an OverlayFS mount:
`sudo mount -t overlay -o lowerdir=/lowerdir,upperdir=/upperdir,workdir=/workdir overlay /merged`
Note that workdir is a required working directory for OverlayFS (for cache). It must be an empty directory and on the same filesystem as upperdir.

eg, mount a data-vol to system-vol
`sudo mount -t overlay -o lowerdir=system-vol,upperdir=data-vol/data,workdir=data-vol/work overlay system-vol`

umount
`sudo umount /merged`
`sudo umount -l system-vol  # Lazy unmount (will unmount when not in use)`
`sudo umount -f system-vol  # Force unmount (use with caution)`

To check which processes are using the mount
`lsof +D system-vol`
```
yisheng@rpi400:/media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground $ sudo lsof +D /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286
lsof: WARNING: can't stat() fuse.gvfsd-fuse file system /run/user/1000/gvfs
      Output information may be incomplete.
lsof: WARNING: can't stat() fuse.portal file system /run/user/1000/doc
      Output information may be incomplete.
COMMAND   PID    USER   FD   TYPE DEVICE SIZE/OFF     NODE NAME
bash     9107 yisheng  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
bash     9555 yisheng  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
bash     9718 yisheng  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
bash    10102 yisheng  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
code    11278 yisheng  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
code    11314 yisheng  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
sudo    12668    root  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
sudo    12669    root  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
lsof    12670    root  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
lsof    12671    root  cwd    DIR    8,0     4096 23199745 /media/yisheng/2243fe50-307e-43ec-b855-e15aa9be2286/playground
```