## Usage

Although this project does not contain a large amount of complicated code, it has not yet solved the problem of being user-friendly and is still under development. This note is intended to help readers understand the project's procedure.

### Steps

#### 1. Provide Cases
This step is crucial for the algorithm to understand the valid interrelationships of the building blocks and their frequency. Different rooms' cases were created manually and separately using a small tool created by the script in `design.py`. In the repository, the cases I have created are already in the folders.

#### Tile Size
The standard tile size is 55cm, which corresponds to the width between the two arms of a person. This measurement is relevant for the depth of a wardrobe, the narrow passage in a room, and is also close to the depth of counters, the sizes of fridges, and washing machines. All items are rounded to 55cm. For example, wall thickness, whether 7cm or 20cm, is rounded to 0. The widths of doors, whether 90cm or 100cm, are all rounded to a module with 2 tile width - 110cm.

#### 2. Make Rooms
`jigsaw.py` in each folder is the main script for generating the rooms. It retains the schemes with the highest scores.

#### 3. Make the Combined Floor Plan
Then, in the 'combine' folder, the script can access all the room schemes and create combined floor plans for a complete unit.

#### 4. Fitness Function
The fitness function (or the evaluation) is included in the main script as polynomial equations. Currently, you have to find it in the script and manually change the coefficients.
