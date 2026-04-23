
local store_dir = "MD_steps"
local store_interval = 1
local store_step_prefix = ""
local store_files = "*fdf *TSHS *TSDE *XV"

local istep_store = 0



function siesta_comm()

    
    -- Initialize the storage directory if this is the beggining of the MD
    if siesta.state == siesta.INIT_MD then
        init_store_dir()
    end

    -- After each step, store the step files
    if siesta.state == siesta.FORCES then
        store_step(istep_store)
        istep_store = istep_store + 1
    end
    

    

end


-- ----------------------------------------------------
--           MD STORAGE HELPER FUNCTIONS
-- ----------------------------------------------------

function init_store_dir()

    if not siesta.IONode then
        -- only allow the IOnode to perform stuff...
        return
    end

    -- Create the directory where the dataset will be stored
    os.execute("mkdir " .. store_dir)

    -- Store the basis
    os.execute("mkdir " ..  store_dir .. "/basis")
    os.execute("cp *.ion* " .. store_dir .. "/basis")
end

function store_step(istep)

    if not siesta.IONode then
        -- only allow the IOnode to perform stuff...
        return
    end

    -- If the step is a multiple of the store interval, store the frame
    if istep % store_interval == 0 then
        os.execute("mkdir " .. store_dir .. "/" .. store_step_prefix .. istep)
        os.execute("cp " .. store_files .. " " .. store_dir .. "/" .. store_step_prefix .. istep)
    end


end


