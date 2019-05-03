### MPI

What happens when a code is running on multiple cores?

The whole piece of code is loaded into all cores that are being used, but it runs differently in different cores.

Each core only takes care of its own task, not executing others, though the whole code is loaded into every core.

By using **if statement**, each core's code can automatically detect whether the **code block** is belonging to it or not. However, it has to be mentioned that those blocks that do not belong to this core are loaded into its memory. They are just not being executed.



### Distribute an array to multiple cores

- we have an array with index from $0$ to $n$
- we have $N$ cores

How to distribute this array to those cores evenly?

$i$th element of that array should go to core
$$
mode(i,N)
$$


### Workflow of MPI

- Every processor loads the code and run line by line 
- Some blocks are for only for specific processors exclusively, then other processors will skip them
- Processors finish their exclusive tasks, then broadcast or send the result to other processors if necessary
- go to next line



It has to be mentioned that every line of the code can be seen by all processors, but they might not be executed by some processors if they are exclusive for their designated processor.

**All processors run the whole piece of program in their own way!**

