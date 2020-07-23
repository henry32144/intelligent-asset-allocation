import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Autocomplete from '@material-ui/lab/Autocomplete';
import TextField from '@material-ui/core/TextField';
import IconButton from '@material-ui/core/IconButton';
import SearchIcon from '@material-ui/icons/Search';
import matchSorter from 'match-sorter'

const useStyles = makeStyles((theme) => ({
  root: {},
  stockComponent: {},
  searchBox: {
    padding: '2px 4px',
    display: 'flex',
    alignItems: 'center',
    minWidth: 330,
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
    <div className={additionalStyles.stockComponent}>
      <Autocomplete
        freeSolo
        className={classes.searchBox}
        id="search-box"
        disableClearable
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