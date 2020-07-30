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
    return matchSorter(options, inputValue, { keys: ['companyName', 'companySymbol'] }).slice(0, 10);
  };

  const stockOnSelected = (event, newValue) => {
    addStockToPortfolio(newValue);
  };

  const addStockToPortfolio = (newValue) => {
    console.log(newValue)
    if (selectedStocks.find(x => x.companyId === newValue.companyId) != null) {
      props.setDialogTitle("Error")
      props.setDialogMessage("The stock is already in the list");
      props.openMessageDialog();
    } else {
      var newPortfolioStocks = selectedStocks.map(function(item, index, array){
        return item.companySymbol;
      });
      setSelectedStocks([...newPortfolioStocks, newValue.companySymbol]);
      props.setDialogTitle("Success")
      props.setDialogMessage("Add " + newValue.companyName + " to your portfolio");
      props.openMessageDialog();
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
        getOptionLabel={(option) => option.companyName}
        filterOptions={filterOptions}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Add company"
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