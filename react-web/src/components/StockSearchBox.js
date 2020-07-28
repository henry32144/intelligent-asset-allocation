import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Autocomplete from '@material-ui/lab/Autocomplete';
import TextField from '@material-ui/core/TextField';
import matchSorter from 'match-sorter'

const useStyles = makeStyles((theme) => ({
  searchBox: {
    display: 'flex',
    alignItems: 'center',
    minWidth: 300,
  },
}));

function StockSearchBox(props) {
  const [value, setValue] = React.useState(null);

  const { additionalStyles, companyData, selectedStocks, setSelectedStocks } = props
  const classes = useStyles();
  const filterOptions = (options, { inputValue }) => {
    return matchSorter(options, inputValue, { keys: ['company_name', 'symbol'] }).slice(0, 10);
  };

  const stockOnSelected = (event, newValue) => {
    if (selectedStocks.find(x => x.companyId === newValue.id_) != null) {
      props.setDialogMessage("The stock is already in the list");
      props.openMessageDialog();
    } else {
      const newSelectedStock = {
        "companyName": newValue.company_name,
        "companySymbol": newValue.symbol,
        "companyId": newValue.id_
      };
      setSelectedStocks([...selectedStocks, newSelectedStock]);
    }
  };


  return (
    <div>
      <Autocomplete
        freeSolo
        className={classes.searchBox}
        id="search-box"
        disableClearable
        size="small"
        onChange={stockOnSelected}
        options={companyData}
        getOptionLabel={(option) => option.company_name}
        filterOptions={filterOptions}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Search company"
            margin="normal"
            variant="outlined"
            InputProps={{ ...params.InputProps, type: 'search' }}
          />
        )}
      />
    </div>
  );
}

export default StockSearchBox;